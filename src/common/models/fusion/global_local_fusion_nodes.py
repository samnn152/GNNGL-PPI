# pyright: reportUnknownMemberType=false
"""Fixed, concatenated, and feature-wise fusion implementations."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FixedSumGlobalLocalFusionNode(nn.Module):
    """Fuse branches by adding equally scaled global and local vectors."""

    def forward(self, global_x: Tensor, local_x: Tensor) -> Tensor:
        """Return the element-wise sum of both branch representations."""
        return global_x + local_x


class ConcatMLPGlobalLocalFusionNode(nn.Module):
    """Learn a joint representation from concatenated branch vectors."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Create the projection from two branch widths back to one."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, global_x: Tensor, local_x: Tensor) -> Tensor:
        """Concatenate matching node vectors and project them through an MLP."""
        return self.mlp(torch.cat([global_x, local_x], dim=-1))


class FeatureWiseGlobalLocalFusionNode(nn.Module):
    """Fuse branches with an independently learned gate per node feature."""

    def __init__(
        self,
        hidden_size: int = 512,
        gate_hidden_size: int = 512,
        initial_alpha: float = 0.5,
        preserve_sum_scale: bool = True,
    ) -> None:
        """Initialize feature-wise gates with the requested starting balance."""
        super().__init__()
        initial_alpha = min(max(initial_alpha, 1e-6), 1 - 1e-6)
        initial_logit = torch.logit(torch.tensor(initial_alpha, dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, hidden_size),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(initial_logit))
        self.preserve_sum_scale = preserve_sum_scale
        self.last_alpha: Tensor | None = None

    @property
    def alpha(self) -> Tensor:
        """Return the latest mean global-branch weight across nodes and features."""
        if self.last_alpha is None:
            return torch.sigmoid(self.gate[-1].bias).mean()
        return self.last_alpha.mean()

    def forward(self, global_x: Tensor, local_x: Tensor) -> Tensor:
        """Blend each feature using a gate inferred from both branch vectors."""
        alpha = torch.sigmoid(self.gate(torch.cat([global_x, local_x], dim=-1)))
        self.last_alpha = alpha.detach()
        fused = alpha * global_x + (1 - alpha) * local_x
        if self.preserve_sum_scale:
            return 2 * fused
        return fused
