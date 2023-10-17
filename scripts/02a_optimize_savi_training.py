import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

from object_centric_video_prediction.lib.config import Config

matplotlib.use('Agg')  # for avoiding memory leak
import torch

from object_centric_video_prediction.data.load_data import unwrap_batch_data
from object_centric_video_prediction.lib.arguments import process_experiment_directory_argument
from object_centric_video_prediction.lib.logger import Logger
from object_centric_video_prediction.lib.visualizations import visualize_decomp, visualize_recons

from object_centric_video_prediction.base.baseTrainer import BaseTrainer


class OptTrainer(BaseTrainer):

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):

        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)

        # forward pass
        videos, targets = videos.to(self.device), targets.to(self.device)
        out_model = self.model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        if inference_only:
            return out_model, None

        # if necessary, doing loss computation, backward pass, optimization, and computing metrics
        self.loss_tracker(
                pred_imgs=reconstruction_history.clamp(0, 1),
                target_imgs=targets.clamp(0, 1)
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_slots"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training_slots"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    @torch.no_grad()
    def visualizations(self, batch_data, epoch, iter_):

        if(iter_ % self.exp_params["training_slots"]["image_log_frequency"] != 0):
            return

        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model
        N = min(10, videos.shape[1])  # max of 10 frames for sleeker figures

        # output reconstructions versus targets
        visualize_recons(
                imgs=targets[0][:N],
                recons=reconstruction_history[0][:N].clamp(0, 1),
                tag="target",
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

        # output reconstructions and input images
        visualize_recons(
                imgs=videos[0][:N],
                recons=reconstruction_history[0][:N].clamp(0, 1),
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

        # Rendered individual objects
        fig, _, _ = visualize_decomp(
                individual_recons_history[0][:N].clamp(0, 1),
                savepath=None,
                tag="objects_decomposed",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        plt.close(fig)

        # Rendered individual object masks
        fig, _, _ = visualize_decomp(
                masks_history[0][:N].clamp(0, 1),
                savepath=None,
                tag="masks",
                cmap="gray",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_,
            )
        plt.close(fig)

        # Rendered individual combination of an object with its masks
        recon_combined = masks_history[0][:N] * individual_recons_history[0][:N]
        fig, _, _ = visualize_decomp(
                recon_combined.clamp(0, 1),
                savepath=None,
                tag="reconstruction_combined",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        plt.close(fig)
        return

    def training_loop(self, trial):

        num_epochs = self.exp_params["training_slots"]["num_epochs"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.model.eval()
            self.valid_epoch(epoch)

            trial.report(100 * self.validation_losses[-1] + obj_mask_error(self.model, self.valid_loader, self.exp_params, self.device, epoch), epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            self.model.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler or lr-warmup
            self.warmup_scheduler(
                    iter=-1,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=True,
                    control_metric=self.validation_losses[-1]
                )

        self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        return 100 * self.validation_losses[-1] + obj_mask_error(self.model, self.valid_loader, self.exp_params, self.device, epoch)


def obj_mask_error(model, valid_loader, exp_params, device, epoch):
    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    error = []

    for i, data in progress_bar:
        videos, targets, _, initializer_kwargs = unwrap_batch_data(exp_params, data)

        # forward pass
        videos, targets = videos.to(device), targets.to(device)
        out_model = model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        error_iter = torch.mean(torch.min(torch.stack([1 - torch.flatten(masks_history), torch.flatten(masks_history)]), dim=0)[0])

        progress_bar.set_description(f"Epoch {epoch + 1} iter {i}: obj mask error {error_iter.item():.5f}. ")

        error.append(torch.mean(error_iter).item())

    return np.mean(error) * exp_params["model"]["SAVi"]["num_slots"]


def define_trial(trial, exp_path):
    cfg = Config(exp_path)
    cfg_dict = cfg.load_exp_config_file()

    cfg_dict["model"]["SAVi"]["num_slots"] = trial.suggest_int("num_slots", 3, 6)
    cfg_dict["model"]["SAVi"]["slot_dim"] = trial.suggest_int("slot_dim", 64, 128, step=32)
    channel_dim_enc = trial.suggest_int("channel_dim_enc", 16, 32, step=8)
    num_channels_enc = trial.suggest_int("num_channels_enc", 3, 5)
    cfg_dict["model"]["SAVi"]["num_channels"] = [channel_dim_enc for _ in range(num_channels_enc)]
    cfg_dict["model"]["SAVi"]["mlp_encoder_dim"] = trial.suggest_int("mlp_encoder_dim", 32, 64, step=16)
    cfg_dict["model"]["SAVi"]["mlp_hidden"] = trial.suggest_int("mlp_hidden", 64, 128, step=32)
    channel_dim_dec = trial.suggest_int("channel_dim_dec", 32, 64, step=16)
    num_channels_dec = trial.suggest_int("num_channels_dec", 3, 5)
    cfg_dict["model"]["SAVi"]["num_channels_decoder"] = [channel_dim_dec for _ in range(num_channels_dec)]
    cfg_dict["model"]["SAVi"]["kernel_size"] = trial.suggest_int("kernel_size", 3, 7, step=2)
    cfg_dict["model"]["SAVi"]["num_iterations_first"] = trial.suggest_int("num_iterations_first", 1, 3)
    cfg_dict["model"]["SAVi"]["num_iterations"] = trial.suggest_int("num_iterations", 1, 3)
    resolution = trial.suggest_int("resolution", 64, 128, step=64)
    cfg_dict["model"]["SAVi"]["resolution"] = [resolution, resolution]
    decoder_resolution = resolution // 2 ** (num_channels_dec - 1)
    cfg_dict["model"]["SAVi"]["decoder_resolution"] = [decoder_resolution, decoder_resolution]
    cfg_dict["model"]["SAVi"]["use_predictor"] = trial.suggest_categorical("use_predictor", [True, False])
    cfg_dict["model"]["SAVi"]["initializer"] = trial.suggest_categorical("initializer", ["Learned", "LearnedRandom"])

    cfg_dict["training_slots"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

    cfg_dict["training_slots"]["batch_size"] = 32 if resolution > 64 else 64
    cfg_dict["training_slots"]["warmup_steps"] = 5000 if resolution > 64 else 2500
    cfg_dict["training_slots"]["scheduler_steps"] = 200000 if resolution > 64 else 100000

    cfg.save_exp_config_file(exp_params=cfg_dict)


def objective(trial):
    exp_path = process_experiment_directory_argument("experiments/vbb_exps/optuna_test")
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting SAVi training procedure", message_type="new_exp")

    define_trial(trial, exp_path)

    trainer = OptTrainer(exp_path=exp_path)
    trainer.setup_model()
    trainer.load_data()
    return trainer.training_loop(trial)


if __name__ == "__main__":
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=100), direction="minimize")
    study.optimize(objective, n_trials=300, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


#
