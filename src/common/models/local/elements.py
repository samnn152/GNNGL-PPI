# pyright: reportUnknownMemberType=false
"""Reusable neural-network elements for local encoding."""
from __future__ import annotations

from typing import Any

from torch import Tensor, nn
import torch.nn.functional as F


class Identity(nn.Module):
    """No-op module used where a configurable transform or norm is disabled."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()

    def forward(self, value: Tensor) -> Tensor:
        """Return ``value`` unchanged."""
        return value

    def reset_parameters(self) -> None:
        """Provide a reset-compatible no-op for composite modules."""
        return None


class MLP(nn.Module):
    """Configurable stack of linear, normalization, and ReLU layers."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_count: int = 2,
        with_final_activation: bool = True,
        with_norm: bool = True,
        bias: bool = True,
        **legacy_names: Any,
    ) -> None:
        """Build a fixed-width projection with optional final activation."""
        super().__init__()
        # Keep the historical nlayer keyword at the boundary only.
        if 'nlayer' in legacy_names:
            layer_count = int(legacy_names.pop('nlayer'))
        if legacy_names:
            raise TypeError(f"Unknown MLP options: {', '.join(legacy_names)}")
        hidden_size = input_size
        self.layers = nn.ModuleList([
            nn.Linear(
                input_size if index == 0 else hidden_size,
                hidden_size if index < layer_count - 1 else output_size,
                bias=(index == layer_count - 1 and not with_final_activation and bias) or not with_norm,
            )
            for index in range(layer_count)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size if index < layer_count - 1 else output_size)
            if with_norm else Identity()
            for index in range(layer_count)
        ])
        self.layer_count = layer_count
        self.with_final_activation = with_final_activation

    def reset_parameters(self) -> None:
        """Reset every linear and normalization layer."""
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform a batch of feature vectors through the configured stack."""
        output = inputs
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            output = layer(output)
            if index < self.layer_count - 1 or self.with_final_activation:
                output = F.relu(norm(output))
        return output
