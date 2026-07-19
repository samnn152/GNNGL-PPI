"""Parse the supported PPI interaction-label vocabulary."""

from typing import ClassVar

class PPILabelSchema:
    """Define the stable multi-label class order used by PPI tensors."""
    CLASS_TO_INDEX: ClassVar[dict[str, int]] = {
        'reaction': 0,
        'binding': 1,
        'ptmod': 2,
        'activation': 3,
        'inhibition': 4,
        'catalysis': 5,
        'expression': 6,
    }
    CLASS_COUNT = 7

    @classmethod
    def empty_label(cls) -> list[int]:
        """Return a zero-filled multi-hot label with the canonical class count."""
        return [0] * cls.CLASS_COUNT
