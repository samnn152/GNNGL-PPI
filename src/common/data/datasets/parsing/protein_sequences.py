"""Load protein sequences from the configured input format."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


class ProteinSequenceStore:
    def __init__(self) -> None:
        self.sequences: dict[str, str] = {}
        self.lengths: list[int] = []

    def load(self, sequence_path: str) -> tuple[dict[str, str], list[int]]:
        self.sequences = {}
        self.lengths = []
        with open(sequence_path) as file:
            for raw_line in tqdm(file):
                protein_id, sequence = raw_line.strip().split('\t')
                if protein_id not in self.sequences:
                    self.sequences[protein_id] = sequence
                    self.lengths.append(len(sequence))
        if not self.lengths:
            raise ValueError(f"No protein sequences found in {sequence_path}")
        print(f"protein num: {len(self.sequences)}")
        print(f"protein average length: {float(np.average(self.lengths))}")
        print(f"protein max & min length: {max(self.lengths)}, {min(self.lengths)}")
        return self.sequences, self.lengths
