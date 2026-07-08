import json
import os
import random

from src.common.utils import GraphSplitSampler


class DatasetSplitter:
    @staticmethod
    def split(ppi_list, edge_num, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if not random_new:
            with open(train_valid_index_path, 'r') as file:
                return json.load(file)

        index_dir = os.path.dirname(train_valid_index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

        if mode == 'random':
            print('+++++++++++++++++++++++++++++random++++++++++++++++++++++++++++++++++++++')
            return DatasetSplitter._random_split(edge_num, test_size, train_valid_index_path)
        if mode in {'bfs', 'dfs'}:
            print("use {} method split train and valid dataset".format(mode))
            return DatasetSplitter._graph_split(ppi_list, edge_num, test_size, mode, train_valid_index_path)

        print("your mode is {}, you should use bfs, dfs or random".format(mode))
        return {}

    @staticmethod
    def _random_split(edge_num, test_size, train_valid_index_path):
        directed_edge_count = int(edge_num // 2)
        edge_indices = [index for index in range(directed_edge_count)]
        random.shuffle(edge_indices)

        split_index = int(directed_edge_count * (1 - test_size))
        split_dict = {
            'train_index': edge_indices[:split_index],
            'valid_index': edge_indices[split_index:],
        }
        DatasetSplitter._write_split(train_valid_index_path, split_dict)
        return split_dict

    @staticmethod
    def _graph_split(ppi_list, edge_num, test_size, mode, train_valid_index_path):
        directed_edge_count = int(edge_num // 2)
        node_to_edge_index = DatasetSplitter._node_to_edge_index(ppi_list, directed_edge_count)
        node_num = len(node_to_edge_index)
        sub_graph_size = int(directed_edge_count * test_size)

        if mode == 'bfs':
            print('+++++++++++++++++++++++++++++bfs++++++++++++++++++++++++++++++++++++++')
            selected_edge_index = GraphSplitSampler.get_bfs_sub_graph(
                ppi_list,
                node_num,
                node_to_edge_index,
                sub_graph_size,
            )
        else:
            print('+++++++++++++++++++++++++++++dfs++++++++++++++++++++++++++++++++++++++')
            selected_edge_index = GraphSplitSampler.get_dfs_sub_graph(
                ppi_list,
                node_num,
                node_to_edge_index,
                sub_graph_size,
            )

        all_edge_index = [index for index in range(directed_edge_count)]
        unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))
        assert len(unselected_edge_index) + len(selected_edge_index) == directed_edge_count

        split_dict = {
            'train_index': unselected_edge_index,
            'valid_index': selected_edge_index,
        }
        DatasetSplitter._write_split(train_valid_index_path, split_dict)
        return split_dict

    @staticmethod
    def _node_to_edge_index(ppi_list, directed_edge_count):
        node_to_edge_index = {}
        for edge_index in range(directed_edge_count):
            source_node, target_node = ppi_list[edge_index]
            node_to_edge_index.setdefault(source_node, []).append(edge_index)
            node_to_edge_index.setdefault(target_node, []).append(edge_index)
        return node_to_edge_index

    @staticmethod
    def _write_split(train_valid_index_path, split_dict):
        with open(train_valid_index_path, 'w') as file:
            file.write(json.dumps(split_dict))
