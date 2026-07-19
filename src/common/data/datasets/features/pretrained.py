"""Load or derive pretrained protein feature vectors."""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

FloatArray = NDArray[np.floating[Any]]


class PretrainedProteinFeatureEncoder:
    """Resolve stored embeddings or deterministic sequence-based fallbacks."""

    @staticmethod
    def pretrained_key(protein_name: str) -> str:
        """Normalize a protein identifier to the key used by embedding files."""
        return protein_name.split('.', 1)[1] if '.' in protein_name else protein_name

    @classmethod
    def load_or_fallback(
        cls,
        pre_emb_path: str,
        sequences: dict[str, str],
    ) -> dict[str, FloatArray]:
        """Load serialized embeddings or derive stable fallback vectors."""
        if pre_emb_path and os.path.exists(pre_emb_path):
            with open(pre_emb_path, 'rb') as file:
                return cast(dict[str, FloatArray], pickle.load(file))
        print(f"Warning: pretrained embedding file not found: {pre_emb_path}")
        print("Using deterministic 512-dim sequence fallback embeddings instead.")
        return {
            cls.pretrained_key(protein_name): cls.sequence_embedding(sequence)
            for protein_name, sequence in sequences.items()
        }

    @staticmethod
    def sequence_embedding(sequence: str, dim: int = 512) -> FloatArray:
        """Encode amino-acid frequencies and hashed k-mers into a fixed vector."""
        embedding = np.zeros(dim, dtype=np.float32)
        if not sequence:
            return embedding
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        amino_acid_index = {acid: index for index, acid in enumerate(amino_acids)}
        for amino_acid in sequence:
            if amino_acid in amino_acid_index:
                embedding[amino_acid_index[amino_acid]] += 1.0
        embedding[:len(amino_acids)] /= max(len(sequence), 1)
        hash_start = len(amino_acids)
        hash_dim = dim - hash_start
        if hash_dim <= 0:
            return embedding
        for kmer_size in (2, 3):
            for sequence_index in range(max(len(sequence) - kmer_size + 1, 0)):
                token = sequence[sequence_index:sequence_index + kmer_size].encode('utf-8')
                digest = hashlib.blake2b(token, digest_size=8).digest()
                bucket = int.from_bytes(digest, 'little') % hash_dim
                embedding[hash_start + bucket] += 1.0 if digest[0] % 2 == 0 else -1.0
        hash_norm = float(np.linalg.norm(embedding[hash_start:]))
        if hash_norm > 0:
            embedding[hash_start:] /= hash_norm
        return embedding

    @classmethod
    def build_protein_features(
        cls,
        protein_names: dict[str, int],
        pretrained_embeddings: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Associate every indexed protein with its normalized pretrained vector."""
        return {
            protein_name: np.asarray(pretrained_embeddings[cls.pretrained_key(protein_name)])
            for protein_name in tqdm(protein_names.keys())
        }
