# pyright: reportUnknownMemberType=false
"""Asymmetric objectives for imbalanced PPI labels."""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


class AsymmetricLoss(nn.Module):
    """Multi-label loss that focuses negatives and positives independently.

    Negative probability clipping and separate focusing powers reduce the
    effect of easy negatives in imbalanced PPI label prediction.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float | None = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        """Configure class-specific focusing, clipping, and numerical stability."""
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return the summed asymmetric multi-label loss for a logits batch."""
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1 - positive_probability
        if self.clip is not None and self.clip > 0:
            negative_probability = (negative_probability + self.clip).clamp(max=1)
        loss = targets * torch.log(positive_probability.clamp(min=self.eps))
        loss += (1 - targets) * torch.log(negative_probability.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            probability = positive_probability * targets + negative_probability * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                weight = torch.pow(1 - probability, gamma)
            loss *= weight
        return -loss.sum()


class AsymmetricLossOptimized(AsymmetricLoss):
    """API-compatible ASL implementation without untyped mutable scratch fields."""

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float | None = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ) -> None:
        """Initialize the optimized variant with the requested ASL parameters."""
        super().__init__(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)


class ASLSingleLabel(nn.Module):
    """Asymmetric focal-style objective for mutually exclusive class labels."""

    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        eps: float = 0.1,
        reduction: Literal['mean', 'sum'] = 'mean',
    ) -> None:
        """Configure positive/negative focusing, smoothing, and reduction."""
        super().__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """Compute smoothed asymmetric loss from class logits and class indices."""
        class_count = inputs.size(-1)
        log_predictions = self.logsoftmax(inputs)
        target_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        anti_targets = 1 - target_classes
        positive_probability = torch.exp(log_predictions) * target_classes
        negative_probability = (1 - torch.exp(log_predictions)) * anti_targets
        weight = torch.pow(
            1 - positive_probability - negative_probability,
            self.gamma_pos * target_classes + self.gamma_neg * anti_targets,
        )
        if self.eps > 0:
            target_classes = target_classes.mul(1 - self.eps).add(self.eps / class_count)
        loss = -(target_classes * log_predictions * weight).sum(dim=-1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()
