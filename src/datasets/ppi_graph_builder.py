import numpy as np
import torch
from torch_geometric.data import Data

from src.common.utils import UnionFindSet


class PPIGraphBuilder:
    @staticmethod
    def connected_components_count(node_num, ppi_list):
        union_find = UnionFindSet(node_num)
        for source_node, target_node in np.array(ppi_list):
            union_find.union(source_node, target_node)
        return union_find

    @classmethod
    def build(cls, node_num, protein_names, ppi_list, ppi_label_list, protein_features, support_features):
        union_find = cls.connected_components_count(node_num, ppi_list)
        print("Connected domain num: {}".format(union_find.count))

        multi_label_types = cls._multi_label_types(ppi_label_list)
        edge_index = torch.tensor(np.array(ppi_list), dtype=torch.long)
        edge_attr = torch.tensor(np.array(ppi_label_list), dtype=torch.long)
        node_features = cls._node_feature_tensor(protein_names, protein_features, dtype=torch.float)
        support_node_features = cls._node_feature_tensor(protein_names, support_features)

        graph_data = Data(
            x=node_features,
            x_origin=support_node_features,
            edge_index=edge_index.T,
            edge_attr_1=edge_attr,
            edge_mul=multi_label_types,
        )
        return graph_data, union_find, edge_index, edge_attr, multi_label_types, node_features, support_node_features

    @staticmethod
    def _multi_label_types(ppi_label_list):
        unique_labels = list(set(tuple(label) for label in ppi_label_list))
        label_to_type = {label: index for index, label in enumerate(unique_labels)}
        return torch.tensor(np.array([label_to_type[tuple(label)] for label in ppi_label_list]))

    @staticmethod
    def _node_feature_tensor(protein_names, feature_dict, dtype=None):
        features = []
        for protein_name, expected_index in protein_names.items():
            assert expected_index == len(features)
            features.append(feature_dict[protein_name])

        feature_array = np.array(features)
        if dtype is None:
            return torch.tensor(feature_array)
        return torch.tensor(feature_array, dtype=dtype)
