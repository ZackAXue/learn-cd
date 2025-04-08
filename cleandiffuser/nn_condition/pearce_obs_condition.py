from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition, get_mask
from cleandiffuser.utils import at_least_ndim


class PearceObsCondition(IdentityCondition):
    """ Observation condition from DiffusionBC: https://arxiv.org/abs/2301.10677

    The model suggests using multi-frame observations as conditions.
    It encodes each frame of observation using the same MLP,
    then flattens them to create a condition embedding.

    Args:
        obs_dim: int,
            The dimension of the observation. Suppose the observation has shape (b, To, obs_dim),
            where b is the batch size, To is the number of frames, and obs_dim is the dimension of each frame.
        emb_dim: int,
            The dimension of the condition embedding. Default: 128
        flatten: bool,
            Whether to flatten the condition embedding. Default: False
        dropout: float,
            The label dropout rate. Default: 0.25

    Examples:
        >>> nn_condition = PearceObsCondition(obs_dim=3, emb_dim=128, flatten=False)
        >>> obs = torch.randn(2, 10, 3)
        >>> nn_condition(obs).shape
        torch.Size([2, 10, 128])
        >>> nn_condition = PearceObsCondition(obs_dim=3, emb_dim=128, flatten=True)
        >>> obs = torch.randn(2, 10, 3)
        >>> nn_condition(obs).shape
        torch.Size([2, 1280])
    """
    def __init__(self, obs_dim: int, emb_dim: int = 128, flatten: bool = False, dropout: float = 0.25):
        super().__init__(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))
        self.flatten = flatten

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        mask = at_least_ndim(get_mask(
            mask, (obs.shape[0],), self.dropout, self.training, obs.device),
            2 if self.flatten else 3)
        embs = self.mlp(obs)  # (b, To, emb_dim)
        embs = torch.flatten(embs, 1) if self.flatten else embs
        return embs * mask


class PearceObsConditionV2(IdentityCondition):
    """
    Modified observation condition from DiffusionBC: https://arxiv.org/abs/2301.10677
    This version allows for different input and output time horizons.
    Args:
        obs_dim: int, The dimension of the observation. Observation shape (b, horizon1, obs_dim)
        emb_dim: int, The dimension of the condition embedding. Default: 128
        in_horizon: int, The input time horizon. Required when out_horizon is specified.
        out_horizon: int, The desired output time horizon (horizon2). Default: None (same as input)
        flatten: bool, Whether to flatten the condition embedding. Default: False
        dropout: float, The label dropout rate. Default: 0.25
    Examples:
    >>> nn_condition = PearceObsConditionV2(obs_dim=3, emb_dim=128, in_horizon=10, out_horizon=15)
    >>> obs = torch.randn(2, 10, 3) # [B, horizon1, D]
    >>> nn_condition(obs).shape
    torch.Size([2, 15, 128]) # [B, horizon2, emb_dim]
    >>> nn_condition = PearceObsConditionV2(obs_dim=3, emb_dim=128, in_horizon=10, out_horizon=15, flatten=True)
    >>> obs = torch.randn(2, 10, 3) # [B, horizon1, D]
    >>> nn_condition(obs).shape
    torch.Size([2, 1920]) # [B, horizon2*emb_dim]
    """
    def __init__(self, obs_dim: int, emb_dim: int = 128, in_horizon: Optional[int] = None,
                 out_horizon: Optional[int] = None, flatten: bool = False, dropout: float = 0.25):
        super().__init__(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.flatten = flatten
        self.out_horizon = out_horizon
        
        # Create the temporal_transform layer with the correct dimensions from the start
        if out_horizon is not None:
            if in_horizon is None:
                raise ValueError("in_horizon must be specified when out_horizon is specified")
            self.temporal_transform = nn.Linear(in_horizon, out_horizon)
        else:
            # If out_horizon is None, we will use the same horizon as the input
            self.temporal_transform = None
            
    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, in_horizon, _ = obs.shape
        
        # Process using MLP as before
        embs = self.mlp(obs)  # (b, horizon1, emb_dim)
        
        # Handle the temporal dimension transformation if out_horizon is specified
        if self.out_horizon is not None:
            embs = embs.transpose(1, 2)  # (b, emb_dim, horizon1)
            
            # Use the pre-created temporal_transform
            embs = self.temporal_transform(embs)  # (b, emb_dim, horizon2)
            embs = embs.transpose(1, 2)  # (b, horizon2, emb_dim)
            
            # Create appropriate mask for the new temporal dimension
            if mask is not None:
                mask = at_least_ndim(get_mask(
                    mask, (batch_size,), self.dropout, self.training, obs.device), 3)
                # Adjust mask to match the new temporal dimension
                if mask.shape[1] != self.out_horizon:
                    new_mask = torch.ones((batch_size, self.out_horizon, 1),
                                         device=mask.device, dtype=mask.dtype)
                    mask = new_mask * mask[:, 0:1, :]  # Broadcast the first mask value
        
        # Handle flattening if requested (after temporal transformation)
        if self.flatten:
            # Now flattening will give us (b, out_horizon*emb_dim)
            embs = torch.flatten(embs, 1)
            mask = at_least_ndim(get_mask(
                mask, (batch_size,), self.dropout, self.training, obs.device), 2)
        else:
            mask = at_least_ndim(get_mask(
                mask, (batch_size,), self.dropout, self.training, obs.device), 3)
        
        return embs * mask