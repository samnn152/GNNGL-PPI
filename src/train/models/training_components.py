"""Construct the concrete components used by the training model."""

import torch
from torch import nn

from src.train.models.components import TrainingComponents
from src.common.device import DeviceResolver
from src.common.models.configuration.gnn import GNNModelConfig
from src.train.models.configuration import TrainConfig
from src.common.models.core import GNNGL_PPI
from src.common.models.losses import AsymmetricLossOptimized


class TrainingComponentFactory:
    """Build the device, model, optimizer, scheduler, and selected loss."""

    def build(self, config: TrainConfig) -> TrainingComponents:
        """Construct all runtime components from a resolved training configuration."""
        device = DeviceResolver.resolve(config.device)
        model = GNNGL_PPI(GNNModelConfig(
            fusion_strategy=config.fusion_strategy,
            feature_source=config.feature_source,
            local_encoder=config.local_encoder,
        )).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                verbose=True,
            )
            if config.use_lr_scheduler else None
        )
        loss: nn.Module = (
            nn.BCEWithLogitsLoss()
            if config.loss_type == 'bce'
            else AsymmetricLossOptimized(
                gamma_neg=1,
                gamma_pos=0,
                clip=0.05,
                disable_torch_grad_focal_loss=True,
            )
        )
        return TrainingComponents(
            device=device,
            model=model,
            optimizer=optimizer,
            loss=loss.to(device),
            scheduler=scheduler,
        )
