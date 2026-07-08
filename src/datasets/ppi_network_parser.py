import copy
import json

from tqdm import tqdm

from src.datasets.ppi_label_schema import PPILabelSchema


class PPINetworkParser:
    def __init__(self, skip_head=True, p1_index=0, p2_index=1, label_index=2):
        self.skip_head = skip_head
        self.p1_index = p1_index
        self.p2_index = p2_index
        self.label_index = label_index

    def parse(self, ppi_path, exclude_protein_path=None, bigger_ppi_path=None, graph_undirection=True):
        excluded_proteins = self._load_excluded_proteins(exclude_protein_path)
        protein_to_index = {}
        edge_to_index = {}
        edge_labels = []

        self._read_ppi_file(ppi_path, excluded_proteins, protein_to_index, edge_to_index, edge_labels)
        if bigger_ppi_path is not None:
            self._read_ppi_file(bigger_ppi_path, {}, protein_to_index, edge_to_index, edge_labels)

        protein_pairs = self._indexed_pairs(edge_to_index)
        original_protein_pairs = copy.deepcopy(protein_pairs)
        numeric_pairs = self._to_numeric_pairs(protein_pairs, protein_to_index)

        if graph_undirection:
            original_edge_count = len(numeric_pairs)
            for edge_index in range(original_edge_count):
                numeric_pairs.append(numeric_pairs[edge_index][::-1])
                edge_labels.append(edge_labels[edge_index])

        return {
            'ppi_list': numeric_pairs,
            'origin_ppi_list': original_protein_pairs,
            'ppi_dict': edge_to_index,
            'ppi_label_list': edge_labels,
            'protein_name': protein_to_index,
            'node_num': len(protein_to_index),
            'edge_num': len(numeric_pairs),
        }

    @staticmethod
    def _load_excluded_proteins(exclude_protein_path):
        if exclude_protein_path is None:
            return {}

        with open(exclude_protein_path, 'r') as file:
            proteins = json.load(file)
        return {protein: index for index, protein in enumerate(proteins)}

    def _read_ppi_file(self, ppi_path, excluded_proteins, protein_to_index, edge_to_index, edge_labels):
        skip_header = self.skip_head
        with open(ppi_path) as file:
            for raw_line in tqdm(file):
                if skip_header:
                    skip_header = False
                    continue

                columns = raw_line.strip().split('\t')
                protein_a = columns[self.p1_index]
                protein_b = columns[self.p2_index]
                label_name = columns[self.label_index]

                if protein_a in excluded_proteins or protein_b in excluded_proteins:
                    continue

                self._register_protein(protein_a, protein_to_index)
                self._register_protein(protein_b, protein_to_index)
                self._register_interaction(protein_a, protein_b, label_name, edge_to_index, edge_labels)

    @staticmethod
    def _register_protein(protein_id, protein_to_index):
        if protein_id not in protein_to_index:
            protein_to_index[protein_id] = len(protein_to_index)

    @staticmethod
    def _interaction_key(protein_a, protein_b):
        left, right = sorted((protein_a, protein_b))
        return "{}__{}".format(left, right)

    def _register_interaction(self, protein_a, protein_b, label_name, edge_to_index, edge_labels):
        interaction_key = self._interaction_key(protein_a, protein_b)
        label_index = PPILabelSchema.CLASS_TO_INDEX[label_name]

        if interaction_key not in edge_to_index:
            edge_to_index[interaction_key] = len(edge_to_index)
            label = PPILabelSchema.empty_label()
            label[label_index] = 1
            edge_labels.append(label)
            return

        existing_label = edge_labels[edge_to_index[interaction_key]]
        existing_label[label_index] = 1

    @staticmethod
    def _indexed_pairs(edge_to_index):
        protein_pairs = []
        for expected_index, interaction_key in enumerate(tqdm(edge_to_index.keys())):
            actual_index = edge_to_index[interaction_key]
            assert actual_index == expected_index
            protein_pairs.append(interaction_key.split('__'))
        return protein_pairs

    @staticmethod
    def _to_numeric_pairs(protein_pairs, protein_to_index):
        numeric_pairs = []
        for protein_a, protein_b in tqdm(protein_pairs):
            numeric_pairs.append([protein_to_index[protein_a], protein_to_index[protein_b]])
        return numeric_pairs
