"""Model configuration for checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.models.configuration.experiment import (
    DatasetType,
    DeviceName,
    FeatureSource,
    FusionStrategy,
    LocalEncoder,
    LossType,
)
from src.common.models.configuration.data import SplitMode


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    dataset_type: DatasetType
    finetune_type: str
    mode: SplitMode
    description: str
    data_dir: str
    pretrain_dir: str
    index_dir: str
    output_dir: str
    ppi_path: str
    pseq_path: str
    vec_path: str
    pre_emb_path: str
    index_path: str
    gnn_model: str
    fusion_strategy: FusionStrategy
    feature_source: FeatureSource
    local_encoder: LocalEncoder
    subgraph_hops: int
    max_subgraph_nodes: int
    max_subgraph_edges: int
    loss_type: LossType
    metrics_csv: str
    test_batch_size: int
    device: DeviceName
    test_all: bool
