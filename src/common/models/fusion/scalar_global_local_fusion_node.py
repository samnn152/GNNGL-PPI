# pyright: reportUnknownMemberType=false
"""Scalar global-local fusion implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ScalarGlobalLocalFusionNode(nn.Module):
    """Fuse every node and feature with one learned global scalar weight."""

    def __init__(self, initial_alpha: float = 0.5, preserve_sum_scale: bool = True) -> None:
        """Initialize the learnable global-local balance in logit space."""
        super().__init__()
        initial_alpha = min(max(initial_alpha, 1e-6), 1 - 1e-6)
        initial_logit = torch.logit(torch.tensor(initial_alpha, dtype=torch.float32))
        self.alpha_logit = nn.Parameter(initial_logit)
        self.preserve_sum_scale = preserve_sum_scale

    @property
    def alpha(self) -> Tensor:
        """Return the constrained global-branch weight in the unit interval."""
        return torch.sigmoid(self.alpha_logit)

    def forward(self, global_x: Tensor, local_x: Tensor) -> Tensor:
        """Blend all branch values using the shared scalar weight."""
        alpha = self.alpha
        fused = alpha * global_x + (1 - alpha) * local_x
        if self.preserve_sum_scale:
            return 2 * fused
        return fused
