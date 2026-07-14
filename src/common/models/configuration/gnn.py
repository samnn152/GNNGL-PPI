"""Shared configuration and request values for GNNGL-PPI models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from src.common.models.ppi_graph import PPIGraph
from src.common.models.configuration.experiment import FeatureSource, FusionStrategy, LocalEncoder


@dataclass(frozen=True, slots=True)
class GNNModelConfig:
    """Architecture choices for the global, local, and fusion model branches."""

    hidden_size: int = 512
    num_layers: int = 1
    use_jk: bool = False
    train_eps: bool = True
    feature_fusion: str | None = None
    class_count: int = 7
    fusion_strategy: FusionStrategy = 'feature_wise'
    feature_source: FeatureSource = 'both'
    local_encoder: LocalEncoder = 'subgraph'


@dataclass(frozen=True, slots=True)
class EdgePredictionBatch:
    """Graph tensors and selected edge identifiers for one forward pass."""

    graph: PPIGraph
    edge_index: Tensor
    edge_ids: list[int] | Tensor


@dataclass(frozen=True, slots=True)
class SubgraphKernelConfig:
    """Hyperparameters controlling bounded local-subgraph representation learning."""
    input_size: int = 512
    output_size: int = 512
    layer_count: int = 1
    dropout: float = 0.2
    hop_embedding_size: int = 16
    bias: bool = False
    residual: bool = True
    pooling: Literal['mean', 'sum', 'max'] = 'mean'
    embeddings: tuple[int, ...] = (0, 1)
    combine_mode: Literal['add', 'concat'] = 'add'
    mlp_layers: int = 1
