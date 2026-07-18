# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
"""Shared node and edge tensors representing a prepared PPI graph."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data


class PPIGraph(Data):
    """Typed PyG data object shared by training and evaluation."""

    x: Tensor  # pyright: ignore[reportIncompatibleMethodOverride] -- PyG exposes Optional[Tensor]
    x_origin: Tensor
    edge_index: Tensor  # pyright: ignore[reportIncompatibleMethodOverride] -- required by this project
    edge_attr_1: Tensor
    edge_mul: Tensor

    train_mask: list[int]
    train_mask_got: list[int]
    val_mask: list[int]
    test_mask: list[int]
    test1_mask: list[int]
    test2_mask: list[int]
    test3_mask: list[int]

    edge_index_got: Tensor
    edge_attr_got: Tensor
    edge_mul_type_got: Tensor

    subgraphs_batch: Tensor
    subgraphs_nodes_mapper: Tensor
    subgraphs_edges_mapper: Tensor
    combined_subgraphs: Tensor
    hop_indicator: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def prepare_for_training(self, device: torch.device) -> None:
        """Release construction-only data, normalize floats, and move devices."""
        if hasattr(self, 'x_origin'):
            del self.x_origin
        for key, value in self.to_dict().items():
            if isinstance(value, Tensor) and value.dtype == torch.float64:
                setattr(self, key, value.float())
        self.to(device)
