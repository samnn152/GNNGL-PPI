"""Typed contracts and result values used by training models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.common.models.ppi_graph import PPIGraph
from src.train.models.context import TrainContext
from src.train.models.contracts import EdgePredictionModel


@dataclass(frozen=True, slots=True)
class EpochStats:
    """Aggregate loss and classification metrics for one epoch phase."""

    loss: float
    recall: float
    precision: float
    f1: float


class TrainObserverProtocol(Protocol):
    """Lifecycle callbacks consumed by ``GNNTrainer`` for progress reporting."""

    suppress_step_logs: bool

    def on_train_start(self, context: object | None, total_epochs: int) -> None:
        """Receive the beginning of a complete training run."""
        ...

    def on_epoch_start(self, epoch: int) -> None:
        """Receive the zero-based index of an epoch about to run."""
        ...

    def on_phase(self, phase: str, message: str) -> None:
        """Receive a fine-grained status update within an epoch."""
        ...

    def on_epoch_end(
        self,
        epoch: int,
        train_stats: EpochStats,
        valid_stats: EpochStats,
        best_valid_f1: float,
        best_valid_epoch: int,
        fusion_alpha: float | None,
    ) -> None:
        """Receive completed metrics, best-validation state, and fusion balance."""
        ...

    def on_train_end(self) -> None:
        """Receive successful completion of the training loop."""
        ...


@dataclass(frozen=True, slots=True)
class TrainingSession:
    """Runtime objects that jointly define one model-training session."""

    model: EdgePredictionModel
    graph: PPIGraph
    loss: nn.Module
    optimizer: Optimizer
    device: torch.device
    observer: TrainObserverProtocol
    scheduler: ReduceLROnPlateau | None = None

    @classmethod
    def from_context(cls, context: TrainContext, observer: TrainObserverProtocol) -> TrainingSession:
        """Extract validated graph and model dependencies from a prepared context."""
        graph = context.require_graph()
        components = context.require_training_components()
        return cls(
            model=components.model,
            graph=graph,
            loss=components.loss,
            optimizer=components.optimizer,
            device=components.device,
            observer=observer,
            scheduler=components.scheduler,
        )


@dataclass(frozen=True, slots=True)
class TrainOptions:
    """Trainer control values separated from heavyweight session objects."""

    result_file_path: str
    save_path: str
    batch_size: int = 512
    epochs: int = 1000
    checkpoint_interval: int = 0
    graph_only_train: bool = False

    @classmethod
    def from_context(cls, context: TrainContext) -> TrainOptions:
        """Create options from resolved arguments and pipeline-generated paths."""
        if context.result_file_path is None or context.save_path is None:
            raise RuntimeError("Train pipeline did not prepare output paths")
        args = context.args
        return cls(
            result_file_path=context.result_file_path,
            save_path=context.save_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            graph_only_train=args.graph_only_train,
        )


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Final metrics, best checkpoint metadata, and resource measurements."""
    train: EpochStats
    valid: EpochStats
    best_valid_f1: float
    best_valid_epoch: int
    save_path: str
    train_time_seconds: float
    model_size_mb: float | None
