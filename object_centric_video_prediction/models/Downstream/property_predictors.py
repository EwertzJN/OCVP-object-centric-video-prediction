"""
Decoder modules for property prediction.

The modules decode the 'frozen' object slots into different object 
properties, which might include:
  - color, shape, mass, position, velocity, charge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import object_centric_video_prediction.lib.hungarian as hungarian


__all__ = ["PropertyPredictor"]


class PropertyPredictor(nn.Module):
    """
    Module that orquestrates the different submodules to predict each of the object
    and motion properties.
    
    Args:
    -----
    datasets: torch.utils.data.Dataset
        Dataset with object properties. we require it to verify which properties it has and 
        the property type and number of values
    properties: list
        List with the names of the properties to predict
    slot_dim: int
        Dimensionality of the object slots
    hidden_dim: int
        Hidden dimensionality of the MLP used to predict the properties
    only_on_last: bool
        If True, the properties are only predicted at the final prediction time step.
    """
    
    PROPERTY_TYPES = ["categorical", "continuous", "temporal"]

    def __init__(self, dataset, properties, slot_dim, hidden_dim, num_context, num_preds,
                 only_on_last=True):
        """ Module initializer """
        super().__init__()
        assert hasattr(dataset, "PROPERTIES"), f"Dataset has not attribute 'PROPERTIES'"
        for prop in properties:
            assert prop in dataset.PROPERTIES, f"{prop = } not in {dataset.PROPERTIES}..."
        self.properties = properties
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.only_on_last = only_on_last
        self.num_context = num_context
        self.num_preds = num_preds
        
        # for each property, we check which type it is and instanciate a corresponding predictor
        self.property_predictors = {}
        for prop in properties:
            prop_data = dataset.PROPERTIES[prop]
            property_predictor = self.instanciate_property_predictor_module(
                prop_type=prop_data["type"],
                num_outputs=prop_data["num_values"],
                prop_name=prop
            )
            self.property_predictors[prop] =  {
                "model": property_predictor,
                "type": prop_data["type"],
                "loss": self._get_loss(prop_data["type"])
            }
        
        return
        
    def instanciate_property_predictor_module(self, prop_type, num_outputs, prop_name):
        """
        Instanciating a property predictor network. This could be:
          - MLP for categorical properties (color, shape, ...) 
          - MLP for some continuous properties (position, charge, ...) 
          - Transformer? for other continuous properties (velocity, ...) 
        """
        assert prop_type in self.PROPERTY_TYPES, f"{prop_type = } not in {self.PROPERTY_TYPES}"
        if prop_type in ["categorical", "continuous"]:
            model = PropertyMLP(
                property_name=prop_name,
                in_dim=self.slot_dim,
                hidden_dim=self.hidden_dim,
                out_dim=num_outputs
            )
        elif prop_type == "temporal":
            raise ValueError("'Temporal' properties are not yet supported for prediction...")
        else:
            raise ValueError(f"{prop_type = } not in {self.PROPERTY_TYPES}")
        return model
        
    def forward(self, slots, targets):
        """
        Forward pass thorough all property predictor modules.
        
        
        Args:
        -----
        slots: torch tensor
            Predicted object slots. Shape is (B, num_preds, num_slots, slot_dim).
        targets: dict
            Dictionary containing the targets for each of the object properties
        """
        # keeping and reshaping desired targets
        targets = self._prepare_targets(
            meta=targets,
            num_context=self.num_context,
            num_preds=self.num_preds
        )
        
        # computing predictions and loss, either for the final step, or aggragating over all pred. steps.
        if self.only_on_last:
            predictions, losses = self.predict_properties(slots, targets, time_step=-1)
        else:
            predictions, losses = {k: [] for k in self.properties}, {k: [] for k in self.properties}
            for i in range(self.num_preds):
                cur_preds, cur_loss = self.predict_properties(slots, targets, time_step=i)
                for k in cur_preds.keys():
                    predictions[k].append(cur_preds[k])
                    losses[k].append(cur_loss[k])
            losses = {k: torch.stack(v).mean() for k, v in losses.items()}
            predictions = {k: torch.stack(v, dim=1) for k, v in predictions.items()}
        return predictions, losses

    def predict_properties(self, slots, targets, time_step):
        """
        Predicting the object properties at a particular time step
        
        The process is as follows:
            1. We predict the property values
            2. We compute pairwise scores between predictions and all targets
            3. We use the Hungarian algorithm to find the best matching between slots and objects
          
        Args:
        -----
        slots: torch tensor
            Predicted object slots. Shape is (B, num_preds, num_slots, slot_dim).
        targets: dict
            Dictionary containing the targets for each of the object properties
        time_step: int
            Time step of the slots used to predict the properties
        """
        B = slots.shape[0]
        predictions = {}
        
        # HACK. Reading num of targets
        first_target = targets[list(targets.keys())[0]]
        num_targets = first_target.shape[-1] if len(first_target.shape) == 3 else first_target.shape[-2]
        
        # predicting properties using models
        scores = []
        for prop, prop_pred_data in self.property_predictors.items():
            prop_type = prop_pred_data["type"]
            model = prop_pred_data["model"]
            
            # forward pass computing pairwise scores between predictions and each potential target
            predicted_props = self.forward_model(model, slots, time_step, prop_type)
            predictions[prop] = predicted_props

            score = self._compute_scores(
                predictions=predicted_props,
                targets=targets[prop][:, time_step],
                prop_type=prop_type
            )
            scores.append(score)

        # aggregating scores from different properties
        aggregated_score = sum([s for s in scores])

        # computing matching IDs and re-sorting based on matches
        matches = hungarian.batch_pairwise_matching(
            dist_tensor=aggregated_score,
            method="hungarian",
            maximize=False
        )
        for prop in predictions.keys():
            for b in range(B):  # TODO: Optimize
                pred_pos_aligned = predictions[prop].clone()
                predictions[prop][b, matches[b, :, -1].long()] = pred_pos_aligned[b, matches[b, :, 0].long()]
            predictions[prop] = predictions[prop][:, :num_targets]
        
        # computing loss for each of the properties
        losses = {}
        for prop in self.property_predictors.keys():
            loss = self.property_predictors[prop]["loss"](
                predictions[prop].flatten(0, 1),
                targets[prop][:, time_step].flatten(0, 1)
            )
            losses[prop] = loss

        return predictions, losses

    def forward_model(self, model, slots, time_step, prop_type):
        """
        Forwarding the slots through the corresponding property predictor model
        """
        if prop_type == "temporal":
            raise NotImplementedError("Temporal properties are not yet supported to be predicted...")
        else:
            prediction = model(slots[:, time_step])
        return prediction

    @torch.no_grad()
    def _compute_scores(self, predictions, targets, prop_type):
        """
        Computing matching scores or costs
        """
        if prop_type == "categorical":
            scores = hungarian.batch_categorical_matching(predictions, targets)
        elif prop_type == "continuous":
            scores = hungarian.batch_pairwise_mse(predictions, targets)
        elif prop_type == "temporal":
            pass
        else:
            raise ValueError(f"{prop_type = } not in {self.PROPERTY_TYPES}")
        return scores

    def _get_loss(self, prop_type):
        """
        Instanciating loss component
        """
        if prop_type == "categorical":
            loss = nn.CrossEntropyLoss()
        elif prop_type == "continuous":
            loss = nn.MSELoss()
        elif prop_type == "temporal":
            loss = nn.MSELoss()
        else:
            raise ValueError(f"{prop_type = } not in {self.PROPERTY_TYPES}")
        return loss
    
    def __str__(self):
        """ For Displaying modules """
        disp = super().__str__()[:-2] + f" (only_on_last={self.only_on_last})"
        for prop_pred in self.property_predictors.values():
            disp = disp + ":\n - " + str(prop_pred["model"])
        return disp
    
    def to(self, *args, **kwargs):
        """ Overriding 'to' function to set all sub-modules to the GPU """
        for prop in self.property_predictors.keys():
            model = self.property_predictors[prop]["model"]
            self.property_predictors[prop]["model"] = model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def parameters(self):
        """
        Making sure we can get the parameters from all the submodules
        """
        pred_data = self.property_predictors
        all_params = []
        for prop in self.property_predictors.keys():
            all_params += [p for _, p in self.property_predictors[prop]["model"].named_parameters()]
        return all_params

    def _prepare_targets(self, meta, num_context, num_preds):
        """
        Extracting the desired target properties from the dataset metadata,
        and reshaping them to the number of prediction time-steps.

        Args:
        -----
        meta: dict
            Properties returned by the dataset
        """
        targets = {}
        for prop in self.properties:
            if prop not in meta:
                raise ValueError(f"{prop = } not in {meta.keys() = }...")
            prop_data = meta[prop]
            
            if self.only_on_last:  # predicting only at the last time step
                if len(prop_data.shape) == 4:
                    cur_target = prop_data[:, num_context+num_preds-1].unsqueeze(1)
                else:
                    cur_target = prop_data.unsqueeze(1)
            else:  # predicting at every time step
                if len(prop_data.shape) == 2:
                    cur_target = prop_data.unsqueeze(1).repeat(1, num_preds, 1)
                elif len(prop_data.shape) == 4:
                    cur_target = prop_data[:, num_context:num_context+num_preds]
                else:
                    raise ValueError(f"Shape of {prop} data {prop_data.shape} has neither 4 nor 2 elements")
            targets[prop] = cur_target

        return targets


class PropertyMLP(nn.Module):
    """
    Simple 2-Layer MLP used for property prediction experiments

    Args:
    -----
    property_name: string
        Name of the property this MLP is supposed to predict
    in_dim: int
        Input dimensionality of the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    out_dim: int
        Output dimensionality of the MLP
    """
    
    def __init__(self, property_name, in_dim, hidden_dim, out_dim):
        """
        Module initializer
        """
        super().__init__()
        self.property = property_name
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        return
    
    def forward(self, x):
        """
        Forward pass
        """
        y = self.mlp(x)
        return y
    
    def __str__(self):
        disp = super().__str__()
        disp = f"{self.property} --> {disp}"
        return disp