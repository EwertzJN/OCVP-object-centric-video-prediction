import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from object_centric_video_prediction.data import unwrap_batch_data


def visualize_property_error(video_sequence, rec_sequence, target_obj, target_unmasked, errors, error_titles, target_obj_inds,
                             savepath=None, tag="sequence", add_title=True, add_axis=False, font_size=11, n_channels=3,
                             titles=None, tb_writer=None, iter=0, **kwargs):
  """ Visualizing a grid with several images/frames """

  n_frames = video_sequence.shape[0]
  n_cols = n_frames
  n_rows = 4 + len(error_titles)

  mosaic = [[] for _ in range(n_rows)]
  for i in range(n_rows):
    for j in range(n_cols):
      if i < 4:
        mosaic[i].append(f"{i}{j}")
      else:
        mosaic[i].append(error_titles[i - 4])

  fig, ax = plt.subplot_mosaic(mosaic, layout="constrained")

  figsize = kwargs.pop("figsize", (3 * n_cols, 3 * n_rows))
  fig.set_size_inches(*figsize)
  if ("suptitle" in kwargs):
    fig.suptitle(kwargs["suptitle"])
    del kwargs["suptitle"]

  ims = []
  fs = []
  for i in range(n_frames):
    row, col = i // n_cols, i % n_cols
    a = ax[f'{row}{col}']
    f = video_sequence[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
    if (n_channels == 1):
      f = f[..., 0]
    im = a.imshow(f, **kwargs)
    ims.append(im)
    fs.append(f)
    if (add_title):
      if (titles is not None):
        cur_title = "" if i >= len(titles) else titles[i]
        a.set_title(cur_title, fontsize=font_size)
      else:
        a.set_title(f"Frame {i}", fontsize=font_size)

    a = ax[f'{row+1}{col}'] if n_rows > 1 else ax[col]
    f = rec_sequence[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
    if (n_channels == 1):
      f = f[..., 0]
    im = a.imshow(f, **kwargs)
    ims.append(im)
    fs.append(f)
    if (add_title):
      if (titles is not None):
        cur_title = "" if i >= len(titles) else titles[i]
        a.set_title(cur_title, fontsize=font_size)
      else:
        a.set_title(f"Reconstruction {i}", fontsize=font_size)

    a = ax[f'{row+2}{col}'] if n_rows > 1 else ax[col]
    f = target_obj[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
    if (n_channels == 1):
      f = f[..., 0]
    im = a.imshow(f, **kwargs)
    ims.append(im)
    fs.append(f)
    if (add_title):
      if (titles is not None):
        cur_title = "" if i >= len(titles) else titles[i]
        a.set_title(cur_title, fontsize=font_size)
      else:
        a.set_title(f"Target Object {i} (slot {target_obj_inds[i]})", fontsize=font_size)

    a = ax[f'{row+3}{col}'] if n_rows > 1 else ax[col]
    f = target_unmasked[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
    if (n_channels == 1):
      f = f[..., 0]
    im = a.imshow(f, **kwargs)
    ims.append(im)
    fs.append(f)
    if (add_title):
      if (titles is not None):
        cur_title = "" if i >= len(titles) else titles[i]
        a.set_title(cur_title, fontsize=font_size)
      else:
        a.set_title(f"Target Unmasked {i} (slot {target_obj_inds[i]})", fontsize=font_size)

  # removing axis
  if (not add_axis):
    for row in range(n_rows):
      for col in range(n_cols):
        if n_cols * row + col < 4 * n_frames:
          a = ax[f'{row}{col}']
          a.set_yticks([])
          a.set_xticks([])

  for i, error in enumerate(error_titles):
    a = ax[error]
    a.plot(errors[i])
    a.set_title(error)
    a.grid()
    a.set_xlim([-0.5, errors.shape[1]-0.5])

  if savepath is not None:
    plt.savefig(savepath)
    if tb_writer is not None:
      img_grid = torch.stack(fs).permute(0, 3, 1, 2)
      tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
  return fig, ax, ims


EXP_NAME = "vbb_exps/ReachSpecificTarget_0to4_Distractors_LargeTargets_DenseReward_DictstateObs_FrontviewViewpointDefaultRobot-dataset"
# PRED_EXP_NAME = "Pred_TF_01_Det"
SAVI_MODEL = "checkpoint_epoch_final.pth"
# CHECKPOINT = "checkpoint_epoch_700.pth"

EXP_PATH = os.path.join("/home/user/ewertzj0/OCVP-object-centric-video-prediction/experiments", EXP_NAME)
assert os.path.exists(EXP_PATH), f"{EXP_PATH = } does not exist"

train_lib = __import__('02_train_savi')

evaluator = train_lib.Trainer(
    exp_path=EXP_PATH,
    checkpoint=SAVI_MODEL,
)
evaluator.exp_params["dataset"]["dataset_name"] = 'PropertiesReachDistinctTarget_4_Distractors_LargeTargets_DenseReward_DictstateObs_FrontviewViewpointDefaultRobot-dataset'
evaluator.exp_params["training_slots"]["batch_size"] = 64
evaluator.exp_params["training_prediction"]["sample_length"] = 25
evaluator.load_data()
evaluator.setup_model()

from object_centric_video_prediction.models.Downstream.property_predictors import PropertyPredictor

model = PropertyPredictor(
    dataset=evaluator.train_loader.dataset,
    properties=["color", "positions"],
    # properties=["positions"],
    slot_dim=128,
    hidden_dim=256,
    only_on_last=False,
    num_context=0,
    num_preds=25
).to(evaluator.device)

print(model)

optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

for episode in range(1, 101):
  loss_values = []

  for i, batch_data in enumerate(tqdm(evaluator.train_loader)):
    videos, _, gt_properties, initializer_kwargs = unwrap_batch_data(evaluator.exp_params, batch_data)
    videos = videos.to(evaluator.device)
    gt_properties = {k: v.type(torch.float32).to(evaluator.device) for k, v in gt_properties.items()}

    # extracting slots
    with torch.no_grad():
      slot_history = evaluator.model.predict_slot_history(videos, num_imgs=25, prior_slots=None, step_offset=0, **initializer_kwargs)

    # predicting object properties
    pred_properties, loss = model(
      slots=slot_history,
      targets=gt_properties
    )

    # opitmization
    loss_val = torch.sum(torch.stack([loss[k] for k in loss.keys()]))
    optim.zero_grad()
    loss_val.backward()
    optim.step()

    loss_values.append(loss_val.item())

  print(f"Episode {episode}: Loss={np.average(loss_values):.4f}")

import object_centric_video_prediction.lib.visualizations as visualizations
import object_centric_video_prediction.lib.hungarian as hungarian
from matplotlib import pyplot as plt

batch_data = next(iter(evaluator.valid_loader))

videos, _, gt_properties, initializer_kwargs = unwrap_batch_data(evaluator.exp_params, batch_data)
videos = videos.to(evaluator.device)
gt_properties = {k: v.to(evaluator.device) for k, v in gt_properties.items()}

shutil.rmtree("semantic_analysis")
os.mkdir("semantic_analysis")

for BATCH_IDX in range(batch_data[0].shape[0]):
  os.mkdir(f"semantic_analysis/{BATCH_IDX}")

  with torch.no_grad():
    out_model = evaluator.model(videos, num_imgs=25, decode=True, **initializer_kwargs)
    slot_history, recons_history, ind_recons_history, masks_history = out_model

  target_color_mses = []
  target_pos_mses = []
  target_obj = []
  target_obj_unmasked = []
  target_obj_inds = []

  for j in range(batch_data[0].shape[1]):
    scores = []

    gt_color = gt_properties["color"][BATCH_IDX].unsqueeze(0)[:, j]
    with torch.no_grad():
      pred_color = model.property_predictors["color"]["model"](slot_history[:, j])[BATCH_IDX].unsqueeze(0)
      score = hungarian.batch_pairwise_mse(pred_color, gt_color)
      scores.append(score)

    gt_pos = gt_properties["positions"][BATCH_IDX].unsqueeze(0)[:, j]
    with torch.no_grad():
      pred_pos = model.property_predictors["positions"]["model"](slot_history[:, j])[BATCH_IDX].unsqueeze(0)
      score = hungarian.batch_pairwise_mse(pred_pos, gt_pos)
      scores.append(score)
      aggregated_score = sum([s for s in scores])
      matches = hungarian.batch_pairwise_matching(
        dist_tensor=aggregated_score,
        method="hungarian",
        maximize=False
      )

    gt_color_aligned = torch.full_like(pred_color, torch.nan, dtype=torch.float64)
    gt_color_aligned[:, matches[:, :, 0].long()] = gt_color[:, matches[:, :, -1].long()]

    gt_pos_aligned = torch.full_like(pred_pos, torch.nan, dtype=torch.float64)
    gt_pos_aligned[:, matches[:, :, 0].long()] = gt_pos[:, matches[:, :, -1].long()]

    db = evaluator.valid_loader.dataset

    gt_color_post = [f"({round(gt_color_aligned[0, i, 0].item() * 255)}, {round(gt_color_aligned[0, i, 1].item() * 255)}, {round(gt_color_aligned[0, i, 2].item() * 255)})" if not torch.isnan(gt_color_aligned[0, i, 0]) else None for i in range(len(gt_color_aligned[0]))]
    pred_color_post = [f"({round(pred_color[0, i, 0].item() * 255)}, {round(pred_color[0, i, 1].item() * 255)}, {round(pred_color[0, i, 2].item() * 255)})" for i in range(len(pred_color[0]))]
    color_mses = [((gt_color_aligned[0, i] - pred_color[0, i])**2).mean().item() for i in range(len(gt_color_aligned[0]))]

    gt_pos_post = [f"({round(gt_pos_aligned[0, i, 0].item() * 10, 1)}, {round(gt_pos_aligned[0, i, 1].item() * 10, 1)}, {round(gt_pos_aligned[0, i, 2].item() * 10, 1)})" if not torch.isnan(gt_pos_aligned[0, i, 0]) else None for i in range(len(gt_pos_aligned[0]))]
    pred_pos_post = [f"({round(pred_pos[0, i, 0].item() * 10, 1)}, {round(pred_pos[0, i, 1].item() * 10, 1)}, {round(pred_pos[0, i, 2].item() * 10, 1)})" for i in range(len(pred_pos[0]))]
    pos_mses = [((gt_pos_aligned[0, i] * 10 - pred_pos[0, i] * 10)**2).mean().item() for i in range(len(gt_pos_aligned[0]))]

    objs = (ind_recons_history * masks_history)[BATCH_IDX:BATCH_IDX+1, j]

    target_obj_index = int(matches[0, (matches[:, :, -1].squeeze() == 0).nonzero().squeeze(), 0])
    target_color_mses.append(((gt_color[0, 0] - pred_color[0, target_obj_index])**2).mean().item())
    target_pos_mses.append(((gt_pos[0, 0] * 10 - pred_pos[0, target_obj_index] * 10)**2).mean().item())
    target_obj_inds.append(target_obj_index)
    target_obj.append(objs[0, target_obj_index])
    target_obj_unmasked.append(ind_recons_history[BATCH_IDX, j, target_obj_index])

    # disp = objs.clone()
    # disp[:, matches[:, :, -1].long()] = objs[:, matches[:, :, 0].long()]
    # disp = disp[:, :6]

    # shape_titles = [f"GT={gt_shape_names[i]}   Pred={pred_shape_names[i]}" for i in range(len(gt_shape_names))]
    # color_titles = [f"GT={gt_color_names[i]}   Pred={pred_color_names[i]}" for i in range(len(gt_color_names))]
    # titles = [s + "\n" + c for s, c in zip(shape_titles, color_titles)]

    # pos_titles = [f"GT={gt_pos_post[i]}  Pred={pred_pos_post[i]} \n MSE={round(mses[i], 4)}" for i in range(len(gt_pos_post))]
    # titles = pos_titles

    color_titles = [f"GT={gt_color_post[i]}  Pred={pred_color_post[i]} \n MSE={round(color_mses[i], 4)}" for i in range(len(gt_color_post))]
    pos_titles = [f"GT={gt_pos_post[i]}  Pred={pred_pos_post[i]} \n MSE={round(pos_mses[i], 4)}" for i in range(len(gt_pos_post))]
    titles = [c + "\n" + p for c, p in zip(color_titles, pos_titles)]

    visualizations.visualize_sequence(
        sequence=torch.cat([objs[0], ind_recons_history[BATCH_IDX, j]]),
        titles=titles,
        n_cols=8,
        figsize=(30, 9)
    )
    plt.savefig(f"semantic_analysis/{BATCH_IDX}/semantic_pred{j}.png")
    plt.close()

  visualize_property_error(videos[BATCH_IDX], recons_history[BATCH_IDX], target_obj, target_obj_unmasked, errors=np.stack([np.array(target_color_mses), np.array(target_pos_mses)]), error_titles=["Target Color MSE", "Target Position MSE"], target_obj_inds=target_obj_inds, savepath=f"semantic_analysis/{BATCH_IDX}/video.png")
  plt.close()
