import hashlib
import os
import pickle

import numpy as np
from tqdm import tqdm


class PretrainedProteinFeatureEncoder:
    @staticmethod
    def pretrained_key(protein_name):
        return protein_name.split('.', 1)[1] if '.' in protein_name else protein_name

    @classmethod
    def load_or_fallback(cls, pre_emb_path, sequences):
        if pre_emb_path and os.path.exists(pre_emb_path):
            with open(pre_emb_path, 'rb') as file:
                return pickle.load(file)

        print("Warning: pretrained embedding file not found: {}".format(pre_emb_path))
        print("Using deterministic 512-dim sequence fallback embeddings instead.")
        return {
            cls.pretrained_key(protein_name): cls.sequence_embedding(sequence)
            for protein_name, sequence in sequences.items()
        }

    @staticmethod
    def sequence_embedding(sequence, dim=512):
        embedding = np.zeros(dim, dtype=np.float32)
        if not sequence:
            return embedding

        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        amino_acid_index = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
        for amino_acid in sequence:
            if amino_acid in amino_acid_index:
                embedding[amino_acid_index[amino_acid]] += 1.0
        embedding[:len(amino_acids)] /= max(len(sequence), 1)

        hash_start = len(amino_acids)
        hash_dim = dim - hash_start
        for kmer_size in (2, 3):
            if len(sequence) < kmer_size:
                continue
            for sequence_index in range(len(sequence) - kmer_size + 1):
                token = sequence[sequence_index:sequence_index + kmer_size].encode('utf-8')
                digest = hashlib.blake2b(token, digest_size=8).digest()
                bucket = int.from_bytes(digest, 'little') % hash_dim
                sign = 1.0 if digest[0] % 2 == 0 else -1.0
                embedding[hash_start + bucket] += sign

        hash_norm = np.linalg.norm(embedding[hash_start:])
        if hash_norm > 0:
            embedding[hash_start:] /= hash_norm
        return embedding

    @classmethod
    def build_protein_features(cls, protein_names, pretrained_embeddings):
        protein_features = {}
        for protein_name in tqdm(protein_names.keys()):
            key = cls.pretrained_key(protein_name)
            protein_features[protein_name] = np.array(pretrained_embeddings[key])
        return protein_features
