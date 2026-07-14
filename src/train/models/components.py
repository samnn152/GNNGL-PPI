"""Runtime components consumed by the training model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.train.models.contracts import EdgePredictionModel


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    """Model-side dependencies produced by the training setup pipeline."""

    device: torch.device
    model: EdgePredictionModel
    optimizer: Optimizer
    loss: nn.Module
    scheduler: ReduceLROnPlateau | None = None
