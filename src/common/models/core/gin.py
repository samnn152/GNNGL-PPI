# pyright: reportMissingTypeStubs=false
"""Typed GIN convolution used by the GNNGL-PPI model."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size


class GINConv(MessagePassing):
    """Small typed GIN convolution used by the project.

    The implementation intentionally relies on ``MessagePassing.propagate``
    instead of overriding its version-sensitive fused sparse hook.
    """

    def __init__(
        self,
        network: Callable[[Tensor], Tensor] | nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize aggregation behavior, epsilon handling, and output network."""
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.network = network
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore epsilon to its configured initial value."""
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:
        """Aggregate neighbor features and apply the configured neural network."""
        x_pair: OptPairTensor = (x, x) if isinstance(x, Tensor) else x
        out = self.propagate(edge_index, x=x_pair, size=size)
        target_x = x_pair[1]
        if target_x is not None:
            out = out + (1 + self.eps) * target_x
        return self.network(out)

    def message(self, x_j: Tensor) -> Tensor:
        """Pass source-node features unchanged into additive aggregation."""
        return x_j

    def __repr__(self) -> str:
        """Return a concise representation containing the configured network."""
        return f"{self.__class__.__name__}(network={self.network})"
