"""Parse the supported PPI interaction-label vocabulary."""

from typing import ClassVar

class PPILabelSchema:
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
        return [0] * cls.CLASS_COUNT
