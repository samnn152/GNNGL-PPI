import numpy as np
from tqdm import tqdm


class ProteinSequenceStore:
    def __init__(self):
        self.sequences = {}
        self.lengths = []

    def load(self, sequence_path):
        self.sequences = {}
        self.lengths = []

        with open(sequence_path) as file:
            for raw_line in tqdm(file):
                protein_id, sequence = raw_line.strip().split('\t')
                if protein_id not in self.sequences:
                    self.sequences[protein_id] = sequence
                    self.lengths.append(len(sequence))

        print("protein num: {}".format(len(self.sequences)))
        print("protein average length: {}".format(np.average(self.lengths)))
        print("protein max & min length: {}, {}".format(np.max(self.lengths), np.min(self.lengths)))
        return self.sequences, self.lengths
