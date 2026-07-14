"""Structural contracts accepted by training models."""

from __future__ import annotations

from typing import Mapping, Protocol

from torch import Tensor

from src.common.models.ppi_graph import PPIGraph


class EdgePredictionModel(Protocol):
    """Model operations required by chunked edge training."""

    def train(self, mode: bool = True) -> object: ...
    def eval(self) -> object: ...
    def state_dict(self) -> Mapping[str, Tensor]: ...
    def encode_nodes(self, x: Tensor, edge_index: Tensor, graph: PPIGraph, p: float = 0.5) -> Tensor: ...
    def decode_edges(self, node_embeddings: Tensor, edge_index: Tensor, edge_ids: list[int] | Tensor) -> Tensor: ...
