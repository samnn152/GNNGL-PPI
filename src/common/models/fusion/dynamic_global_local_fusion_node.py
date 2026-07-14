# pyright: reportUnknownMemberType=false
"""Node-wise dynamic global-local fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DynamicGlobalLocalFusionNode(nn.Module):
    """Fuse global and local vectors with one learned gate per node."""

    def __init__(
        self,
        hidden_size: int = 512,
        gate_hidden_size: int = 128,
        initial_alpha: float = 0.5,
        preserve_sum_scale: bool = True,
    ) -> None:
        """Initialize a node-wise gate with the requested starting balance."""
        super().__init__()
        initial_alpha = min(max(initial_alpha, 1e-6), 1 - 1e-6)
        initial_logit = torch.logit(torch.tensor(initial_alpha, dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(initial_logit))
        self.preserve_sum_scale = preserve_sum_scale
        self.last_alpha: Tensor | None = None

    @property
    def alpha(self) -> Tensor:
        """Return the latest mean global-branch weight."""
        if self.last_alpha is None:
            return torch.sigmoid(self.gate[-1].bias.mean())
        return self.last_alpha.mean()

    def forward(self, global_x: Tensor, local_x: Tensor) -> Tensor:
        """Blend matching node vectors using gates derived from both branches."""
        gate_input = torch.cat([global_x, local_x], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))
        self.last_alpha = alpha.detach()
        fused = alpha * global_x + (1 - alpha) * local_x
        if self.preserve_sum_scale:
            return 2 * fused
        return fused
