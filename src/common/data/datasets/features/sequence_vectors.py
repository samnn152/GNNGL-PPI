"""Encode amino-acid sequences using support vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

FloatArray = NDArray[np.floating[Any]]


class SequenceVectorEncoder:
    """Build fixed-length protein matrices from per-amino-acid vectors."""

    def __init__(self, max_len: int = 2000) -> None:
        """Store the maximum number of residues retained for each protein."""
        self.max_len = max_len

    def vectorize(
        self,
        sequences: dict[str, str],
        vec_path: str,
    ) -> tuple[dict[str, FloatArray], int, dict[str, FloatArray]]:
        """Load amino-acid vectors and encode all supplied protein sequences."""
        amino_acid_vectors, vector_dim = self._load_amino_acid_vectors(vec_path)
        print(f"acid vector dimension: {vector_dim}")
        protein_vectors: dict[str, FloatArray] = {}
        for protein_id in tqdm(sequences.keys()):
            sequence_matrix = np.asarray([amino_acid_vectors[acid] for acid in sequences[protein_id]])
            protein_vectors[protein_id] = self.pad_or_trim(sequence_matrix, vector_dim)
        return amino_acid_vectors, vector_dim, protein_vectors

    @staticmethod
    def _load_amino_acid_vectors(vec_path: str) -> tuple[dict[str, FloatArray], int]:
        """Parse a tab-separated amino-acid vector file and infer its dimension."""
        amino_acid_vectors: dict[str, FloatArray] = {}
        vector_dim: int | None = None
        with open(vec_path) as file:
            for raw_line in file:
                columns = raw_line.strip().split('\t')
                vector = np.asarray([float(value) for value in columns[1].split()])
                amino_acid_vectors[columns[0]] = vector
                vector_dim = vector_dim or len(vector)
        if vector_dim is None:
            raise ValueError(f"No amino-acid vectors found in {vec_path}")
        return amino_acid_vectors, vector_dim

    def pad_or_trim(self, sequence_matrix: FloatArray, vector_dim: int) -> FloatArray:
        """Resize a sequence matrix to ``max_len`` without changing feature width."""
        if len(sequence_matrix) > self.max_len:
            return sequence_matrix[:self.max_len]
        if len(sequence_matrix) < self.max_len:
            padding = np.zeros((self.max_len - len(sequence_matrix), vector_dim))
            return np.concatenate((sequence_matrix, padding))
        return sequence_matrix
