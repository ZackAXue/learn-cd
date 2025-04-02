from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from copy import deepcopy
from .basic import DiffusionModel

SUPPORTED_SOLVERS = [
    "ddpm"
    # , 
    # "ddim",
    # "ode_dpmsolver_1", "ode_dpmsolver++_1", "ode_dpmsolver++_2M",
    # "sde_dpmsolver_1", "sde_dpmsolver++_1", "sde_dpmsolver++_2M",
    ]


def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma


class BaseDiffusionSDE(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            # The mask should be in the shape of `x_shape`.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            epsilon: float = 1e-3,

            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min

    @property
    def supported_solvers(self):
        return SUPPORTED_SOLVERS

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):
        raise NotImplementedError

    def loss(self, x0, condition=None, **kwargs):

        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2
        
        loss = loss * self.loss_weight * (1 - self.fix_mask)
        
        # find weighted_regression_tensor in kwargs
        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss *= weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss = self.loss(x0, condition, **kwargs)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)
                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred

        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)

        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(self, *args, **kwargs):
        raise NotImplementedError
# ------------------------------------------------------------------------------

# ------------------------ JDM Continuous Diffusion SDE ------------------------

class JdmContinuousDiffusionSDE(BaseDiffusionSDE):
    """Joint Discrete-Motion Continuous-time Diffusion SDE
    
    This class implements a hierarchical diffusion model for joint discrete action
    and continuous motion planning. It consists of two sub-diffusion models:
    1. High-level (HL) diffusion for generating discrete symbolic actions (represented as bits)
    2. Low-level (LL) diffusion for generating continuous motion trajectories
    
    The two diffusion models are trained jointly in an end-to-end manner, with
    conditioning mechanisms between them to ensure coherence.
        Args:
    - nn_diffusion: BaseNNDiffusion
        The neural network backbone for the Diffusion model.
    - nn_condition: Optional[BaseNNCondition]
        The neural network backbone for the condition embedding.
        
    - fix_mask: Union[list, np.ndarray, torch.Tensor]
        Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        The mask should be in the shape of `x_shape`.
    - loss_weight: Union[list, np.ndarray, torch.Tensor]
        Add loss weight. The weight should be in the shape of `x_shape`.
        
    - classifier: Optional[BaseClassifier]
        Add a classifier to enable classifier-guidance.
        
    - grad_clip_norm: Optional[float]
        Gradient clipping norm.
    - ema_rate: float
        Exponential moving average rate.
    - optim_params: Optional[dict]
        Optimizer parameters.
        
    - epsilon: float
        The minimum time step for the diffusion reverse process. 
        In practice, using a very small value instead of `0` can avoid numerical instability.
        
    - noise_schedule: Union[str, Dict[str, Callable]]
        The noise schedule for the diffusion process. Can be "linear" or "cosine".
    - noise_schedule_params: Optional[dict]
        The parameters for the noise schedule.
        
    - x_max: Optional[torch.Tensor]
        The maximum value for the input data. `None` indicates no constraint.
    - x_min: Optional[torch.Tensor]
        The minimum value for the input data. `None` indicates no constraint.
        
    - predict_noise: bool
        Whether to predict the noise or the data.
        
    - device: Union[torch.device, str]
        The device to run the model.
    """
    def __init__(
            self,
            # ----------------- Neural Networks ----------------- #
            # High-level diffusion for discrete symbolic actions
            nn_diffusion_hl: BaseNNDiffusion,
            # Low-level diffusion for continuous motion
            nn_diffusion_ll: BaseNNDiffusion,
            # Condition networks
            nn_condition_hl_to_ll: Optional[BaseNNCondition] = None,  # y_t-1 conditions x_t
            nn_condition_ll_to_hl: Optional[BaseNNCondition] = None,  # x_t conditions y_t-1
            
            # ----------------- Masks ----------------- #
            # Fix masks for inpainting (init and goal states/coordinates)
            fix_mask_hl: Union[list, np.ndarray, torch.Tensor] = None,  # for discrete actions
            fix_mask_ll: Union[list, np.ndarray, torch.Tensor] = None,  # for motion data
            # Loss weights
            loss_weight_hl: Union[list, np.ndarray, torch.Tensor] = None,
            loss_weight_ll: Union[list, np.ndarray, torch.Tensor] = None,
            
            # ------------------ Plugins ---------------- #
            # Classifiers for guidance (optional)
            classifier_hl: Optional[BaseClassifier] = None,
            classifier_ll: Optional[BaseClassifier] = None,
            
            # ------------------ Hierarchical Parameters ---------------- #
            steps_per_action: int = 6,  # Number of motion steps per symbolic action
            
            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params_hl: Optional[dict] = None,
            optim_params_ll: Optional[dict] = None,
            
            # ------------------- Diffusion Params ------------------- #
            epsilon: float = 1e-3,
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,
            
            # ------------------- Data Constraints ------------------- #
            x_max_hl: Optional[torch.Tensor] = None,  # Max value for discrete actions
            x_min_hl: Optional[torch.Tensor] = None,  # Min value for discrete actions
            x_max_ll: Optional[torch.Tensor] = None,  # Max value for motion data
            x_min_ll: Optional[torch.Tensor] = None,  # Min value for motion data
            
            predict_noise: bool = True,
            device: Union[torch.device, str] = "cpu"
    ):
        # Initialize the base class with HL diffusion parameters
        # We'll initialize only one of the diffusion models with the base class
        super().__init__(
            nn_diffusion_hl, nn_condition_ll_to_hl, fix_mask_hl, loss_weight_hl, 
            classifier_hl, grad_clip_norm, ema_rate, optim_params_hl,
            epsilon, noise_schedule, noise_schedule_params, x_max_hl, x_min_hl, 
            predict_noise, device
        )
        
        # Store hierarchical parameters
        self.steps_per_action = steps_per_action
        
        # ==================== Create a second diffusion model for LL ====================
        # Store LL neural networks
        self.nn_diffusion_ll = nn_diffusion_ll
        self.nn_condition_hl_to_ll = nn_condition_hl_to_ll
        
        # Create LL model and optimizer
        self.model_ll = {"diffusion": self.nn_diffusion_ll, "condition": self.nn_condition_hl_to_ll}
        self.model_ll_ema = {"diffusion": deepcopy(self.nn_diffusion_ll), "condition": deepcopy(self.nn_condition_hl_to_ll) if self.nn_condition_hl_to_ll is not None else None}
        
        # Move LL models to device
        for k, v in self.model_ll.items():
            if v is not None:
                self.model_ll[k] = v.to(self.device)
        for k, v in self.model_ll_ema.items():
            if v is not None:
                self.model_ll_ema[k] = v.to(self.device)
        
        # Set up LL optimizer
        optim_params_ll = optim_params_ll or {}
        params_ll = []
        for k, v in self.model_ll.items():
            if v is not None:
                params_ll.extend(list(v.parameters()))
        lr_ll = optim_params_ll.pop("lr", 1e-4)
        self.optimizer_ll = torch.optim.AdamW(params_ll, lr=lr_ll, **optim_params_ll)
        
        # Set up LL mask and constraints
        self.fix_mask_ll = torch.zeros(1) if fix_mask_ll is None else torch.as_tensor(fix_mask_ll, device=self.device)
        self.loss_weight_ll = torch.ones(1) if loss_weight_ll is None else torch.as_tensor(loss_weight_ll, device=self.device)
        self.x_max_ll = x_max_ll
        self.x_min_ll = x_min_ll
        
        # Set up LL classifier
        self.classifier_ll = classifier_ll
        
        # ==================== Rename parameters for clarity ====================
        # Rename HL parameters for clarity
        self.model_hl = self.model
        self.model_hl_ema = self.model_ema
        self.optimizer_hl = self.optimizer
        self.fix_mask_hl = self.fix_mask
        self.loss_weight_hl = self.loss_weight
        self.x_max_hl = self.x_max
        self.x_min_hl = self.x_min
        self.classifier_hl = self.classifier
        
        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]
            
        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")
    
    # ==================== Training: Score Matching ======================

    def add_noise(self, x0_hl, x0_ll, t_hl=None, t_ll=None, eps_hl=None, eps_ll=None):
        """
        Add noise to both high-level (HL) discrete actions and low-level (LL) motion data.
        
        Args:
            x0_hl: torch.Tensor
                Clean high-level discrete actions, shape (batch_size, horizon, bit_dim)
            x0_ll: torch.Tensor
                Clean low-level motion data, shape (batch_size, horizon*steps_per_action, 3)
            t_hl: Optional[torch.Tensor]
                Time steps for HL diffusion. If None, sampled randomly.
            t_ll: Optional[torch.Tensor]
                Time steps for LL diffusion. If None, sampled randomly.
            eps_hl: Optional[torch.Tensor]
                Noise for HL diffusion. If None, sampled randomly.
            eps_ll: Optional[torch.Tensor]
                Noise for LL diffusion. If None, sampled randomly.
                
        Returns:
            xt_hl: torch.Tensor
                Noisy high-level discrete actions
            t_hl: torch.Tensor
                Time steps used for HL diffusion
            eps_hl: torch.Tensor
                Noise used for HL diffusion
            xt_ll: torch.Tensor
                Noisy low-level motion data
            t_ll: torch.Tensor
                Time steps used for LL diffusion
            eps_ll: torch.Tensor
                Noise used for LL diffusion
        """
        batch_size = x0_hl.shape[0]
        
        # Sample random time steps if not provided
        if t_hl is None:
            t_hl = torch.rand((batch_size,), device=self.device) * \
                (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]
        
        if t_ll is None:
            # Option 1: Use same time steps for LL (simpler)
            t_ll = t_hl
            # Option 2: Sample independent time steps (more flexible)
            # t_ll = torch.rand((batch_size,), device=self.device) * \
            #        (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]
        
        # Sample random noise if not provided
        eps_hl = torch.randn_like(x0_hl) if eps_hl is None else eps_hl
        eps_ll = torch.randn_like(x0_ll) if eps_ll is None else eps_ll
        
        # Add noise to HL discrete actions
        alpha_hl, sigma_hl = self.noise_schedule_funcs["forward"](t_hl, **(self.noise_schedule_params or {}))
        alpha_hl = at_least_ndim(alpha_hl, x0_hl.dim())
        sigma_hl = at_least_ndim(sigma_hl, x0_hl.dim())
        
        xt_hl = alpha_hl * x0_hl + sigma_hl * eps_hl
        xt_hl = (1. - self.fix_mask_hl) * xt_hl + self.fix_mask_hl * x0_hl
        
        # Add noise to LL motion data
        alpha_ll, sigma_ll = self.noise_schedule_funcs["forward"](t_ll, **(self.noise_schedule_params or {}))
        alpha_ll = at_least_ndim(alpha_ll, x0_ll.dim())
        sigma_ll = at_least_ndim(sigma_ll, x0_ll.dim())
        
        xt_ll = alpha_ll * x0_ll + sigma_ll * eps_ll
        xt_ll = (1. - self.fix_mask_ll) * xt_ll + self.fix_mask_ll * x0_ll
        
        return xt_hl, t_hl, eps_hl, xt_ll, t_ll, eps_ll
    
     # ==================== Training ======================

    def loss(self, x0_hl, x0_ll, condition_hl=None, condition_ll=None, **kwargs):
        """
        Calculate the combined loss for both HL and LL diffusion models.
        
        Args:
            x0_hl: torch.Tensor
                Clean high-level discrete actions
            x0_ll: torch.Tensor
                Clean low-level motion data
            condition_hl: Optional
                External condition for HL diffusion
            condition_ll: Optional
                External condition for LL diffusion
            **kwargs: Additional arguments
                
        Returns:
            loss_hl: torch.Tensor
                Loss for HL diffusion
            loss_ll: torch.Tensor
                Loss for LL diffusion
        """
        # Add noise to both HL and LL data
        xt_hl, t_hl, eps_hl, xt_ll, t_ll, eps_ll = self.add_noise(x0_hl, x0_ll)
        
        # Process conditions
        # For HL diffusion, condition could include LL information using cross-attention
        condition_hl_embed = self.model_hl["condition"](condition_hl) if condition_hl is not None else None
        
        # For LL diffusion, condition could include HL information using cross-attention
        condition_ll_embed = self.model_ll["condition"](condition_ll) if condition_ll is not None else None
        
        # Calculate losses for both diffusions
        if self.predict_noise:
            loss_hl = (self.model_hl["diffusion"](xt_hl, t_hl, condition_hl_embed) - eps_hl) ** 2
            loss_ll = (self.model_ll["diffusion"](xt_ll, t_ll, condition_ll_embed) - eps_ll) ** 2
        else:
            loss_hl = (self.model_hl["diffusion"](xt_hl, t_hl, condition_hl_embed) - x0_hl) ** 2
            loss_ll = (self.model_ll["diffusion"](xt_ll, t_ll, condition_ll_embed) - x0_ll) ** 2
        
        # Apply masks and weights
        loss_hl = loss_hl * self.loss_weight_hl * (1 - self.fix_mask_hl)
        loss_ll = loss_ll * self.loss_weight_ll * (1 - self.fix_mask_ll)
        
        # Apply additional weighting if provided
        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss_hl *= weighted_regression_tensor.unsqueeze(-1)
            loss_ll *= weighted_regression_tensor.unsqueeze(-1)
        
        return loss_hl.mean(), loss_ll.mean()

    def update(self, x0_hl, x0_ll, condition_hl=None, condition_ll=None, update_ema=True, 
            hl_weight=1.0, ll_weight=1.0, update_both=True, **kwargs):
        """
        One-step gradient update for both HL and LL diffusion models.
        
        Args:
            x0_hl: torch.Tensor
                Clean high-level discrete actions
            x0_ll: torch.Tensor
                Clean low-level motion data
            condition_hl: Optional
                External condition for HL diffusion
            condition_ll: Optional
                External condition for LL diffusion
            update_ema: bool
                Whether to update the exponential moving average models
            hl_weight: float
                Weight for HL loss in the combined loss
            ll_weight: float
                Weight for LL loss in the combined loss
            update_both: bool
                Whether to update both models or just one of them
            **kwargs: Additional arguments
                
        Returns:
            log: dict
                The log dictionary containing losses and gradient norms
        """
        # Calculate losses
        loss_hl, loss_ll = self.loss(x0_hl, x0_ll, condition_hl, condition_ll, **kwargs)
        
        # Combine losses with weights
        combined_loss = hl_weight * loss_hl + ll_weight * loss_ll
        
        # Backward pass
        combined_loss.backward()
        
        # Update models based on update_both flag
        grad_norm_hl = None
        grad_norm_ll = None
        
        if update_both or hl_weight > 0:
            grad_norm_hl = nn.utils.clip_grad_norm_(
                [p for k, v in self.model_hl.items() if v is not None for p in v.parameters()], 
                self.grad_clip_norm) if self.grad_clip_norm else None
            self.optimizer_hl.step()
            self.optimizer_hl.zero_grad()
        
        if update_both or ll_weight > 0:
            grad_norm_ll = nn.utils.clip_grad_norm_(
                [p for k, v in self.model_ll.items() if v is not None for p in v.parameters()], 
                self.grad_clip_norm) if self.grad_clip_norm else None
            self.optimizer_ll.step()
            self.optimizer_ll.zero_grad()
        
        # Update EMA models if needed
        if update_ema:
            self.ema_update()
        
        # Create log dictionary
        log = {
            "loss_hl": loss_hl.item(),
            "loss_ll": loss_ll.item(),
            "loss_combined": combined_loss.item(),
            "grad_norm_hl": grad_norm_hl,
            "grad_norm_ll": grad_norm_ll
        }
        
        return log

    def ema_update(self):
        """
        Update exponential moving average for both HL and LL models.
        Not Sure Correctness
        """
        # Update HL EMA model
        for k, v in self.model_hl.items():
            if v is not None:
                for param, param_ema in zip(v.parameters(), self.model_hl_ema[k].parameters()):
                    param_ema.data = self.ema_rate * param_ema.data + (1 - self.ema_rate) * param.data
        
        # Update LL EMA model
        for k, v in self.model_ll.items():
            if v is not None:
                for param, param_ema in zip(v.parameters(), self.model_ll_ema[k].parameters()):
                    param_ema.data = self.ema_rate * param_ema.data + (1 - self.ema_rate) * param.data

    def update_classifiers(self, x0_hl, x0_ll, condition_hl, condition_ll):
        """
        Update both classifiers if they exist.
        Not implemented yet.
        """
        log = {}
        
        # Update HL classifier if it exists
        if self.classifier_hl is not None:
            xt_hl, t_hl, _ = self.add_noise(x0_hl, None)[:3]  # Only get HL noise components
            log.update({"classifier_hl": self.classifier_hl.update(xt_hl, t_hl, condition_hl)})
        
        # Update LL classifier if it exists
        if self.classifier_ll is not None:
            _, _, _, xt_ll, t_ll, _ = self.add_noise(None, x0_ll)  # Only get LL noise components
            log.update({"classifier_ll": self.classifier_ll.update(xt_ll, t_ll, condition_ll)})
        
        return log

    # ==================== Sampling: Solving SDE/ODE ======================
    def prepare_hl_condition(self, xt_hl):
        """
        Prepare high-level condition for low-level diffusion.
        This method can be implemented to transform HL actions into a suitable format
        for conditioning the LL model.
        
        Args:
            xt_hl: High-level action sequence
            
        Returns:
            Processed condition tensor
        """
        # Default implementation: return as is
        # Override this method with specific processing as needed
        return xt_hl
    
    def prepare_ll_condition(self, xt_ll):
        """
        Prepare low-level condition for high-level diffusion.
        This method can be implemented to transform LL motion trajectories into a suitable format
        for conditioning the HL model.
        
        Args:
            xt_ll: Low-level motion sequence
            
        Returns:
            Processed condition tensor
        """
        # Default implementation: return as is
        # Override this method with specific processing as needed
        return xt_ll
    
    def sample(
        self,
        # ---------- the known fixed portions ---------- #
        prior_hl: torch.Tensor,    # Fixed portions of high-level actions (e.g., initial states)
        prior_ll: torch.Tensor,    # Fixed portions of low-level motions (e.g., initial coordinates)
        # ----------------- sampling ----------------- #
        solver: str = "ddpm",      # Sampling solver
        n_samples: int = 1,        # Number of samples to generate
        sample_steps: int = 20,    # Number of sampling steps
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        use_ema: bool = True,      # Whether to use EMA models
        temperature: float = 1.0,  # Sampling temperature
        # ------------------ guidance ------------------ #
        w_cfg_hl: float = 0.0,     # Weight for classifier-free guidance (HL)
        w_cfg_ll: float = 0.0,     # Weight for classifier-free guidance (LL)
        # ------------------ others ------------------ #
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        """Hierarchical sampling for joint discrete-motion diffusion.
        
        This method implements the hierarchical sampling process where 
        high-level discrete actions and low-level motion trajectories are 
        generated in an interleaved manner, with conditioning between them.
        
        The conditional relationship follows:
        - High-level: p(y_t-1|y_t, x_t, y_cond)
        - Low-level: p(x_t-1|x_t, y_t-1, x_cond)
        
        Inputs:
        - prior_hl: torch.Tensor
            The known fixed portion of high-level actions. Shape: (n_samples, action_horizon, bit_dim)
        - prior_ll: torch.Tensor
            The known fixed portion of low-level motions. Shape: (n_samples, motion_horizon, motion_dim)
        
        - solver: str
            The solver for the reverse process. Currently only "ddpm" is fully implemented.
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average models.
        - temperature: float
            The temperature for sampling.
        
        - w_cfg_hl: float
            Weight for classifier-free guidance for the high-level model.
        - w_cfg_ll: float
            Weight for classifier-free guidance for the low-level model.
            
        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.
            
        Outputs:
        - x0_hl: torch.Tensor
            Generated high-level action samples.
        - x0_ll: torch.Tensor
            Generated low-level motion samples.
        - log: dict
            The log dictionary containing sampling history and metrics.
        """
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."
        
        # ===================== Initialization =====================
        log = {
            "sample_history_hl": np.empty((n_samples, sample_steps + 1, *prior_hl.shape[1:])) if preserve_history else None,
            "sample_history_ll": np.empty((n_samples, sample_steps + 1, *prior_ll.shape[1:])) if preserve_history else None,
        }
        
        # Select models (EMA or regular)
        model_hl = self.model_hl_ema if use_ema else self.model_hl
        model_ll = self.model_ll_ema if use_ema else self.model_ll
        
        # Move priors to device
        prior_hl = prior_hl.to(self.device)
        prior_ll = prior_ll.to(self.device)
        
        # Initialize with random noise
        xt_hl = torch.randn_like(prior_hl) * temperature
        xt_ll = torch.randn_like(prior_ll) * temperature
        
        # Apply fixed masks (for known portions like initial states)
        xt_hl = xt_hl * (1. - self.fix_mask_hl) + prior_hl * self.fix_mask_hl
        xt_ll = xt_ll * (1. - self.fix_mask_ll) + prior_ll * self.fix_mask_ll
        
        # Store initial state in history if needed
        if preserve_history:
            log["sample_history_hl"][:, 0] = xt_hl.cpu().numpy()
            log["sample_history_ll"][:, 0] = xt_ll.cpu().numpy()
        
        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")
        
        # Compute alphas and sigmas for noise schedule
        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        # For DDPM solver, we need to compute stds
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()
        
        # ===================== Denoising Loop ========================
        with torch.set_grad_enabled(requires_grad):
            for i in reversed(range(1, sample_steps + 1)):
                # Current time step tensor
                t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)
                
                # Step 1: Prepare cross-conditioning - LL conditions HL
                # Here we use the current xt_ll to condition the denoising of yt_hl
                condition_ll_to_hl = self.prepare_ll_condition(xt_ll) if hasattr(self, "prepare_ll_condition") else xt_ll
                condition_vec_ll_to_hl = model_hl["condition"](condition_ll_to_hl) if model_hl["condition"] is not None else None
                
                # Step 2: Denoise HL with LL conditioning
                # This gives us yt-1_hl conditioned on yt_hl and xt_ll
                # TODO: if we want to add cf, _ will be log_p_hl
                pred_hl, _ = self.guided_sampling(
                    xt_hl, t, alphas[i], sigmas[i],
                    model_hl, condition_vec_ll_to_hl, w_cfg_hl, None, 0.0, requires_grad)
                # If we want to clip the prediction, we can add it here
                # pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

                # Transform to eps_theta or x_theta based on predict_noise setting
                if self.predict_noise:
                    eps_theta_hl = pred_hl
                    x_theta_hl = epstheta_to_xtheta(xt_hl, alphas[i], sigmas[i], pred_hl)
                else:
                    x_theta_hl = pred_hl
                    eps_theta_hl = xtheta_to_epstheta(xt_hl, alphas[i], sigmas[i], pred_hl)
                
                # One-step update for HL (DDPM sampling)
                # TODO: if want to add DDIM, we can add it here
                xt_hl_prev = (
                    (alphas[i - 1] / alphas[i]) * (xt_hl - sigmas[i] * eps_theta_hl) +
                    (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta_hl
                )
                
                # Add noise if not the last step
                if i > 1:
                    xt_hl_prev += stds[i] * torch.randn_like(xt_hl)
                
                # Step 3: Prepare cross-conditioning - HL conditions LL
                # Here we use the denoised yt-1_hl to condition the denoising of xt_ll
                condition_hl_to_ll = self.prepare_hl_condition(xt_hl_prev) if hasattr(self, "prepare_hl_condition") else xt_hl_prev
                condition_vec_hl_to_ll = model_ll["condition"](condition_hl_to_ll) if model_ll["condition"] is not None else None
                
                # Step 4: Denoise LL with HL conditioning
                # This gives us xt-1_ll conditioned on xt_ll and yt-1_hl
                pred_ll, _ = self.guided_sampling(
                    xt_ll, t, alphas[i], sigmas[i],
                    model_ll, condition_vec_hl_to_ll, w_cfg_ll, None, 0.0, requires_grad)
                
                # Transform to eps_theta or x_theta based on predict_noise setting
                if self.predict_noise:
                    eps_theta_ll = pred_ll
                    x_theta_ll = epstheta_to_xtheta(xt_ll, alphas[i], sigmas[i], pred_ll)
                else:
                    x_theta_ll = pred_ll
                    eps_theta_ll = xtheta_to_epstheta(xt_ll, alphas[i], sigmas[i], pred_ll)
                
                # One-step update for LL (DDPM sampling)
                xt_ll_prev = (
                    (alphas[i - 1] / alphas[i]) * (xt_ll - sigmas[i] * eps_theta_ll) +
                    (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta_ll
                )
                
                # Add noise if not the last step
                if i > 1:
                    xt_ll_prev += stds[i] * torch.randn_like(xt_ll)
                
                # Apply fixed masks (maintain known portions)
                xt_hl_prev = xt_hl_prev * (1. - self.fix_mask_hl) + prior_hl * self.fix_mask_hl
                xt_ll_prev = xt_ll_prev * (1. - self.fix_mask_ll) + prior_ll * self.fix_mask_ll
                
                # Update current state
                xt_hl = xt_hl_prev
                xt_ll = xt_ll_prev
                
                # Store history if needed
                if preserve_history:
                    log["sample_history_hl"][:, sample_steps - i + 1] = xt_hl.cpu().numpy()
                    log["sample_history_ll"][:, sample_steps - i + 1] = xt_ll.cpu().numpy()
        
        # ================= Post-processing =================
        # TODO: If we want to add cf, add lopg here

        # Clip values to min/max if needed
        if self.x_min_hl is not None or self.x_max_hl is not None:
            xt_hl = xt_hl.clip(self.x_min_hl, self.x_max_hl)
        if self.x_min_ll is not None or self.x_max_ll is not None:
            xt_ll = xt_ll.clip(self.x_min_ll, self.x_max_ll)
        
        return xt_hl, xt_ll, log
    
    # ==================== Saving and Loading ======================  

    def save(self, path: str):
        """
        Save both HL and LL models and their EMA versions to a file.
        
        Args:
            path: str
                Path to save the model checkpoint
        """
        # Create a dictionary with all model parameters
        checkpoint = {
            # HL models
            "model_hl": {k: v.state_dict() if v is not None else None for k, v in self.model_hl.items()},
            "model_hl_ema": {k: v.state_dict() if v is not None else None for k, v in self.model_hl_ema.items()},
            # LL models
            "model_ll": {k: v.state_dict() if v is not None else None for k, v in self.model_ll.items()},
            "model_ll_ema": {k: v.state_dict() if v is not None else None for k, v in self.model_ll_ema.items()},
            # Optimizer states
            "optimizer_hl": self.optimizer_hl.state_dict(),
            "optimizer_ll": self.optimizer_ll.state_dict(),
            # Configuration parameters
            "config": {
                "ema_rate": self.ema_rate,
                "predict_noise": self.predict_noise,
                "epsilon": self.epsilon,
                "steps_per_action": self.steps_per_action,
                "noise_schedule_params": self.noise_schedule_params,
            }
        }
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str, load_optimizers: bool = True):
        """
        Load both HL and LL models and their EMA versions from a file.
        
        Args:
            path: str
                Path to the model checkpoint
            load_optimizers: bool
                Whether to load optimizer states (set to False when loading for inference only)
        """
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load HL models
        for k, v in self.model_hl.items():
            if v is not None and checkpoint["model_hl"][k] is not None:
                v.load_state_dict(checkpoint["model_hl"][k])
        
        for k, v in self.model_hl_ema.items():
            if v is not None and checkpoint["model_hl_ema"][k] is not None:
                v.load_state_dict(checkpoint["model_hl_ema"][k])
        
        # Load LL models
        for k, v in self.model_ll.items():
            if v is not None and checkpoint["model_ll"][k] is not None:
                v.load_state_dict(checkpoint["model_ll"][k])
        
        for k, v in self.model_ll_ema.items():
            if v is not None and checkpoint["model_ll_ema"][k] is not None:
                v.load_state_dict(checkpoint["model_ll_ema"][k])
        
        # Load optimizer states if requested
        if load_optimizers:
            if "optimizer_hl" in checkpoint:
                self.optimizer_hl.load_state_dict(checkpoint["optimizer_hl"])
            if "optimizer_ll" in checkpoint:
                self.optimizer_ll.load_state_dict(checkpoint["optimizer_ll"])
        
        # Load configuration parameters if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            #TODO: 
            # You can selectively load configuration parameters if needed
            # For example:
            # self.ema_rate = config.get("ema_rate", self.ema_rate)
            # self.predict_noise = config.get("predict_noise", self.predict_noise)
        
        print(f"Model loaded from {path}")

    def save_checkpoint(self, path: str, step: int, metrics: dict = None):
        """
        Save a training checkpoint including models, optimizers, training step, and metrics.
        
        Args:
            path: str
                Base path for the checkpoint
            step: int
                Current training step
            metrics: dict
                Dictionary of metrics to save with the checkpoint
        """
        # Create checkpoint directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a dictionary with all model parameters and training state
        checkpoint = {
            # HL models
            "model_hl": {k: v.state_dict() if v is not None else None for k, v in self.model_hl.items()},
            "model_hl_ema": {k: v.state_dict() if v is not None else None for k, v in self.model_hl_ema.items()},
            # LL models
            "model_ll": {k: v.state_dict() if v is not None else None for k, v in self.model_ll.items()},
            "model_ll_ema": {k: v.state_dict() if v is not None else None for k, v in self.model_ll_ema.items()},
            # Optimizer states
            "optimizer_hl": self.optimizer_hl.state_dict(),
            "optimizer_ll": self.optimizer_ll.state_dict(),
            # Training state
            "step": step,
            "metrics": metrics,
            # Configuration parameters
            "config": {
                "ema_rate": self.ema_rate,
                "predict_noise": self.predict_noise,
                "epsilon": self.epsilon,
                "steps_per_action": self.steps_per_action,
                "noise_schedule_params": self.noise_schedule_params,
            }
        }
        
        # Save the checkpoint
        checkpoint_path = f"{path}_step{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint for easy resumption
        latest_path = f"{path}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved to {checkpoint_path} and {latest_path}")

    def load_checkpoint(self, path: str):
        """
        Load a training checkpoint including models, optimizers, training step, and metrics.
        
        Args:
            path: str
                Path to the checkpoint file
                
        Returns:
            step: int
                The training step of the loaded checkpoint
            metrics: dict
                Dictionary of metrics saved with the checkpoint
        """
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load HL models
        for k, v in self.model_hl.items():
            if v is not None and checkpoint["model_hl"][k] is not None:
                v.load_state_dict(checkpoint["model_hl"][k])
        
        for k, v in self.model_hl_ema.items():
            if v is not None and checkpoint["model_hl_ema"][k] is not None:
                v.load_state_dict(checkpoint["model_hl_ema"][k])
        
        # Load LL models
        for k, v in self.model_ll.items():
            if v is not None and checkpoint["model_ll"][k] is not None:
                v.load_state_dict(checkpoint["model_ll"][k])
        
        for k, v in self.model_ll_ema.items():
            if v is not None and checkpoint["model_ll_ema"][k] is not None:
                v.load_state_dict(checkpoint["model_ll_ema"][k])
        
        # Load optimizer states
        self.optimizer_hl.load_state_dict(checkpoint["optimizer_hl"])
        self.optimizer_ll.load_state_dict(checkpoint["optimizer_ll"])
        
        # Get training state
        step = checkpoint.get("step", 0)
        metrics = checkpoint.get("metrics", {})
        
        print(f"Checkpoint loaded from {path}, resuming from step {step}")
        
        return step, metrics
    
# ==================== Training and Evaluation ======================

    def train(self):
        """
        Set all models to training mode.
        """
        # Set HL models to training mode
        for k, v in self.model_hl.items():
            if v is not None:
                v.train()
        
        # Set LL models to training mode
        for k, v in self.model_ll.items():
            if v is not None:
                v.train()
        
        # Set classifiers to training mode if they exist
        if self.classifier_hl is not None:
            self.classifier_hl.model.train()
        if self.classifier_ll is not None:
            self.classifier_ll.model.train()

    def eval(self):
        """
        Set all models to evaluation mode.
        """
        # Set HL models to evaluation mode
        for k, v in self.model_hl.items():
            if v is not None:
                v.eval()
        
        # Set LL models to evaluation mode
        for k, v in self.model_ll.items():
            if v is not None:
                v.eval()
        
        # Set classifiers to evaluation mode if they exist
        if self.classifier_hl is not None:
            self.classifier_hl.model.eval()
        if self.classifier_ll is not None:
            self.classifier_ll.model.eval()