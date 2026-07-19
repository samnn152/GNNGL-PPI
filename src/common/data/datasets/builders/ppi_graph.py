"""Build tensor graphs from parsed PPI records and protein features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.common.utils import UnionFindSet
from src.common.models.ppi_graph import PPIGraph


class PPIGraphBuilder:
    """Convert parsed interactions and feature dictionaries into a tensor graph."""

    @staticmethod
    def connected_components_count(node_num: int, ppi_list: list[list[int]]) -> UnionFindSet:
        """Build a union-find structure describing PPI connected components."""
        union_find = UnionFindSet(node_num)
        for source_node, target_node in np.array(ppi_list):
            union_find.union(source_node, target_node)
        return union_find

    @classmethod
    def build(
        cls,
        node_num: int,
        protein_names: dict[str, int],
        ppi_list: list[list[int]],
        ppi_label_list: list[list[int]],
        protein_features: dict[str, NDArray[np.floating[Any]]],
        support_features: dict[str, NDArray[np.floating[Any]]],
    ) -> tuple[PPIGraph, UnionFindSet, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Create the graph and return it together with its intermediate tensors."""
        union_find = cls.connected_components_count(node_num, ppi_list)
        print("Connected domain num: {}".format(union_find.count))

        multi_label_types = cls._multi_label_types(ppi_label_list)
        edge_index = torch.tensor(np.array(ppi_list), dtype=torch.long)
        edge_attr = torch.tensor(np.array(ppi_label_list), dtype=torch.long)
        node_features = cls._node_feature_tensor(protein_names, protein_features, dtype=torch.float)
        support_node_features = cls._node_feature_tensor(protein_names, support_features)

        graph_data = PPIGraph(
            x=node_features,
            x_origin=support_node_features,
            edge_index=edge_index.T,
            edge_attr_1=edge_attr,
            edge_mul=multi_label_types,
        )
        return graph_data, union_find, edge_index, edge_attr, multi_label_types, node_features, support_node_features

    @staticmethod
    def _multi_label_types(ppi_label_list: list[list[int]]) -> Tensor:
        """Map each distinct multi-hot edge label to a compact integer type."""
        unique_labels = list(set(tuple(label) for label in ppi_label_list))
        label_to_type = {label: index for index, label in enumerate(unique_labels)}
        return torch.tensor(np.array([label_to_type[tuple(label)] for label in ppi_label_list]))

    @staticmethod
    def _node_feature_tensor(
        protein_names: dict[str, int],
        feature_dict: dict[str, NDArray[np.floating[Any]]],
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Stack protein features in the exact order of their assigned node indices."""
        features: list[NDArray[np.floating[Any]]] = []
        for protein_name, expected_index in protein_names.items():
            assert expected_index == len(features)
            features.append(feature_dict[protein_name])

        feature_array = np.stack(features)
        if dtype is None:
            return torch.tensor(feature_array)
        return torch.tensor(feature_array, dtype=dtype)
