# pyright: reportUnknownMemberType=false
"""Shared device selection for training and evaluation."""

from collections.abc import Callable
from typing import ClassVar

import torch

from src.common.models.configuration.experiment import DeviceName

DeviceProbe = tuple[Callable[[], bool], str]


class DeviceResolver:
    """Resolve explicit device names or probe accelerated backends for ``auto``."""

    EXPLICIT_DEVICES: ClassVar[dict[DeviceName, str]] = {
        'cpu': 'cpu',
        'cuda': 'cuda:0',
        'mps': 'mps',
    }
    AUTO_CANDIDATES: ClassVar[tuple[DeviceProbe, ...]] = (
        (torch.cuda.is_available, 'cuda:0'),
        (torch.backends.mps.is_available, 'mps'),
    )

    @classmethod
    def resolve(cls, requested: DeviceName) -> torch.device:
        """Return the requested device, preferring CUDA then MPS for ``auto``."""
        selected = cls.EXPLICIT_DEVICES.get(requested) or next(
            (name for available, name in cls.AUTO_CANDIDATES if available()),
            'cpu',
        )
        return torch.device(selected)
