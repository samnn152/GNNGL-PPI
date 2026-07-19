from __future__ import annotations

"""Shared model configuration for local subgraph extraction."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparseSubgraphConfig:
    """Bounds and hop depth used for sparse ego-subgraph extraction."""
    hops: int = 1
    max_nodes: int = 64
    max_edges: int = 256
