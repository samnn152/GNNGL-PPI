"""Configuration model for a training experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from src.common.models.configuration.data import SplitMode
from src.common.models.configuration.experiment import (
    DatasetType,
    DeviceName,
    FeatureSource,
    FusionStrategy,
    LocalEncoder,
    LossType,
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Complete immutable configuration for one training run."""

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
    go_onehot_path: str | None
    pro_go_def_path: str | None
    split_new: bool
    split_mode: SplitMode
    validation_size: float
    test_size: float
    train_valid_index_path: str
    use_lr_scheduler: bool
    save_path: str
    graph_only_train: bool
    fusion_strategy: FusionStrategy
    feature_source: FeatureSource
    local_encoder: LocalEncoder
    loss_type: LossType
    subgraph_hops: int
    max_subgraph_nodes: int
    max_subgraph_edges: int
    metrics_csv: str
    batch_size: int
    epochs: int
    interactive_ui: bool
    device: DeviceName
    checkpoint_interval: int

    def as_dict(self) -> dict[str, object]:
        """Return a serializable copy for logs and configuration files."""
        return asdict(self)
