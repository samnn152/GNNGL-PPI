"""Parse file-backed PPI networks into typed records."""

from __future__ import annotations

import copy
import json
from collections.abc import Collection
from typing import cast

from tqdm import tqdm

from src.common.models.configuration.data import GNNDataConfig, ParsedPPINetwork
from src.common.data.datasets.parsing.label_schema import PPILabelSchema


class PPINetworkParser:
    """Parse one or more PPI files into indexed proteins and multi-label edges."""

    def __init__(self, config: GNNDataConfig) -> None:
        """Store the column layout and file paths used during parsing."""
        self.config = config

    def parse(self) -> ParsedPPINetwork:
        """Read configured interactions and return a fully indexed network record."""
        config = self.config
        excluded_proteins = self._load_excluded_proteins(config.exclude_protein_path)
        protein_to_index: dict[str, int] = {}
        edge_to_index: dict[str, int] = {}
        edge_labels: list[list[int]] = []

        self._read_ppi_file(config.ppi_path, excluded_proteins, protein_to_index, edge_to_index, edge_labels)
        if config.bigger_ppi_path is not None:
            self._read_ppi_file(config.bigger_ppi_path, set(), protein_to_index, edge_to_index, edge_labels)

        protein_pairs = self._indexed_pairs(edge_to_index)
        original_protein_pairs = copy.deepcopy(protein_pairs)
        numeric_pairs = self._to_numeric_pairs(protein_pairs, protein_to_index)

        if config.undirected:
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
    def _load_excluded_proteins(exclude_protein_path: str | None) -> set[str]:
        """Load an optional JSON list of proteins that parsing must ignore."""
        if exclude_protein_path is None:
            return set()
        with open(exclude_protein_path, 'r') as file:
            proteins = cast(list[str], json.load(file))
        return set(proteins)

    def _read_ppi_file(
        self,
        ppi_path: str,
        excluded_proteins: Collection[str],
        protein_to_index: dict[str, int],
        edge_to_index: dict[str, int],
        edge_labels: list[list[int]],
    ) -> None:
        """Merge valid interactions from one PPI file into parser accumulators."""
        skip_header = self.config.skip_header
        with open(ppi_path) as file:
            for raw_line in tqdm(file):
                if skip_header:
                    skip_header = False
                    continue
                columns = raw_line.strip().split('\t')
                protein_a = columns[self.config.protein_a_column]
                protein_b = columns[self.config.protein_b_column]
                label_name = columns[self.config.label_column]
                if protein_a in excluded_proteins or protein_b in excluded_proteins:
                    continue
                self._register_protein(protein_a, protein_to_index)
                self._register_protein(protein_b, protein_to_index)
                self._register_interaction(protein_a, protein_b, label_name, edge_to_index, edge_labels)

    @staticmethod
    def _register_protein(protein_id: str, protein_to_index: dict[str, int]) -> None:
        """Assign the next stable node index to a previously unseen protein."""
        if protein_id not in protein_to_index:
            protein_to_index[protein_id] = len(protein_to_index)

    @staticmethod
    def _interaction_key(protein_a: str, protein_b: str) -> str:
        """Create an order-independent key for an undirected protein pair."""
        left, right = sorted((protein_a, protein_b))
        return f"{left}__{right}"

    def _register_interaction(
        self,
        protein_a: str,
        protein_b: str,
        label_name: str,
        edge_to_index: dict[str, int],
        edge_labels: list[list[int]],
    ) -> None:
        """Create or update the multi-hot label for one protein interaction."""
        interaction_key = self._interaction_key(protein_a, protein_b)
        label_index = PPILabelSchema.CLASS_TO_INDEX[label_name]
        if interaction_key not in edge_to_index:
            edge_to_index[interaction_key] = len(edge_to_index)
            label = PPILabelSchema.empty_label()
            label[label_index] = 1
            edge_labels.append(label)
            return
        edge_labels[edge_to_index[interaction_key]][label_index] = 1

    @staticmethod
    def _indexed_pairs(edge_to_index: dict[str, int]) -> list[list[str]]:
        """Restore protein identifier pairs in their assigned edge order."""
        protein_pairs: list[list[str]] = []
        for expected_index, interaction_key in enumerate(tqdm(edge_to_index.keys())):
            assert edge_to_index[interaction_key] == expected_index
            protein_pairs.append(interaction_key.split('__'))
        return protein_pairs

    @staticmethod
    def _to_numeric_pairs(
        protein_pairs: list[list[str]],
        protein_to_index: dict[str, int],
    ) -> list[list[int]]:
        """Replace protein identifiers with their numeric node indices."""
        return [
            [protein_to_index[protein_a], protein_to_index[protein_b]]
            for protein_a, protein_b in tqdm(protein_pairs)
        ]
