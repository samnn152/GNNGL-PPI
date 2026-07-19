"""Load or generate PPI edge partitions."""

from __future__ import annotations

import json
import os
import random
from typing import cast

from src.common.utils import GraphSplitSampler
from src.common.models.configuration.data import DatasetSplit, DatasetSplitConfig


class DatasetSplitter:
    """Load persisted edge splits or generate random/BFS/DFS partitions."""

    @staticmethod
    def split(ppi_list: list[list[int]], edge_num: int, config: DatasetSplitConfig) -> DatasetSplit:
        """Return the requested persisted or newly generated dataset split."""
        if not config.regenerate:
            with open(config.index_path, 'r') as file:
                return cast(DatasetSplit, json.load(file))

        index_dir = os.path.dirname(config.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

        if config.mode == 'random':
            print('+++++++++++++++++++++++++++++random++++++++++++++++++++++++++++++++++++++')
            return DatasetSplitter._random_split(edge_num, config)
        print(f"use {config.mode} method split train and valid dataset")
        return DatasetSplitter._graph_split(ppi_list, edge_num, config)

    @staticmethod
    def _random_split(edge_num: int, config: DatasetSplitConfig) -> DatasetSplit:
        """Randomly partition original directed edges by configured ratios."""
        directed_edge_count = edge_num // 2
        edge_indices = list(range(directed_edge_count))
        random.shuffle(edge_indices)
        validation_count = int(directed_edge_count * config.validation_size)
        test_count = int(directed_edge_count * config.test_size)
        train_count = directed_edge_count - validation_count - test_count
        validation_end = train_count + validation_count
        split_dict: DatasetSplit = {
            'train_index': edge_indices[:train_count],
            'valid_index': edge_indices[train_count:validation_end],
            'test_index': edge_indices[validation_end:],
        }
        DatasetSplitter._write_split(config.index_path, split_dict)
        return split_dict

    @staticmethod
    def _graph_split(
        ppi_list: list[list[int]],
        edge_num: int,
        config: DatasetSplitConfig,
    ) -> DatasetSplit:
        """Reserve a connected BFS/DFS edge subset for validation."""
        if config.test_size > 0:
            raise ValueError("Three-way splitting currently supports random mode only")
        directed_edge_count = edge_num // 2
        node_to_edge_index = DatasetSplitter._node_to_edge_index(ppi_list, directed_edge_count)
        sub_graph_size = int(directed_edge_count * config.validation_size)
        if config.mode == 'bfs':
            selected_edge_index = GraphSplitSampler.get_bfs_sub_graph(
                ppi_list, len(node_to_edge_index), node_to_edge_index, sub_graph_size
            )
        else:
            selected_edge_index = GraphSplitSampler.get_dfs_sub_graph(
                ppi_list, len(node_to_edge_index), node_to_edge_index, sub_graph_size
            )
        unselected_edge_index = list(set(range(directed_edge_count)).difference(selected_edge_index))
        assert len(unselected_edge_index) + len(selected_edge_index) == directed_edge_count
        split_dict: DatasetSplit = {
            'train_index': unselected_edge_index,
            'valid_index': selected_edge_index,
            'test_index': [],
        }
        DatasetSplitter._write_split(config.index_path, split_dict)
        return split_dict

    @staticmethod
    def _node_to_edge_index(
        ppi_list: list[list[int]],
        directed_edge_count: int,
    ) -> dict[int, list[int]]:
        """Build the node-to-original-edge adjacency used by graph sampling."""
        node_to_edge_index: dict[int, list[int]] = {}
        for edge_index in range(directed_edge_count):
            source_node, target_node = ppi_list[edge_index]
            node_to_edge_index.setdefault(source_node, []).append(edge_index)
            node_to_edge_index.setdefault(target_node, []).append(edge_index)
        return node_to_edge_index

    @staticmethod
    def _write_split(index_path: str, split_dict: DatasetSplit) -> None:
        """Persist an edge split as JSON for reproducible reuse."""
        with open(index_path, 'w') as file:
            json.dump(split_dict, file)
