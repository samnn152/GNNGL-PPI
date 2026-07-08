import numpy as np
from tqdm import tqdm


class SequenceVectorEncoder:
    def __init__(self, max_len=2000):
        self.max_len = max_len

    def vectorize(self, sequences, vec_path):
        amino_acid_vectors, vector_dim = self._load_amino_acid_vectors(vec_path)
        print("acid vector dimension: {}".format(vector_dim))

        protein_vectors = {}
        for protein_id in tqdm(sequences.keys()):
            sequence_matrix = np.array([amino_acid_vectors[acid] for acid in sequences[protein_id]])
            protein_vectors[protein_id] = self._pad_or_trim(sequence_matrix, vector_dim)
        return amino_acid_vectors, vector_dim, protein_vectors

    @staticmethod
    def _load_amino_acid_vectors(vec_path):
        amino_acid_vectors = {}
        vector_dim = None
        with open(vec_path) as file:
            for raw_line in file:
                columns = raw_line.strip().split('\t')
                vector = np.array([float(value) for value in columns[1].split()])
                amino_acid_vectors[columns[0]] = vector
                if vector_dim is None:
                    vector_dim = len(vector)
        return amino_acid_vectors, vector_dim

    def _pad_or_trim(self, sequence_matrix, vector_dim):
        if len(sequence_matrix) > self.max_len:
            return sequence_matrix[:self.max_len]
        if len(sequence_matrix) < self.max_len:
            padding_len = self.max_len - len(sequence_matrix)
            return np.concatenate((sequence_matrix, np.zeros((padding_len, vector_dim))))
        return sequence_matrix
