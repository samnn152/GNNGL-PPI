"""Structural contracts accepted by training models."""

from __future__ import annotations

from typing import Mapping, Protocol

from torch import Tensor

from src.common.models.ppi_graph import PPIGraph


class EdgePredictionModel(Protocol):
    """Model operations required by chunked edge training."""

    def train(self, mode: bool = True) -> object:
        """Switch the model into the requested training mode."""
        ...

    def eval(self) -> object:
        """Switch the model into evaluation mode."""
        ...

    def state_dict(self) -> Mapping[str, Tensor]:
        """Return the parameters serialized into training checkpoints."""
        ...

    def encode_nodes(self, x: Tensor, edge_index: Tensor, graph: PPIGraph, p: float = 0.5) -> Tensor:
        """Encode graph nodes once for a complete train or validation phase."""
        ...

    def decode_edges(self, node_embeddings: Tensor, edge_index: Tensor, edge_ids: list[int] | Tensor) -> Tensor:
        """Decode selected edges from previously calculated node embeddings."""
        ...
