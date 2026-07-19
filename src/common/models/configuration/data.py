"""Shared configuration models for PPI data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


SplitMode = Literal['random', 'bfs', 'dfs']


class DatasetSplit(TypedDict):
    """Persisted edge indices for train, validation, and test partitions."""
    train_index: list[int]
    valid_index: list[int]
    test_index: list[int]


class ParsedPPINetwork(TypedDict):
    """Indexed interaction data produced by the PPI parser."""
    ppi_list: list[list[int]]
    origin_ppi_list: list[list[str]]
    ppi_dict: dict[str, int]
    ppi_label_list: list[list[int]]
    protein_name: dict[str, int]
    node_num: int
    edge_num: int


@dataclass(frozen=True, slots=True)
class GNNDataConfig:
    """Immutable input-file and parsing options for PPI data preparation."""
    ppi_path: str
    exclude_protein_path: str | None = None
    max_sequence_length: int = 2000
    skip_header: bool = True
    protein_a_column: int = 0
    protein_b_column: int = 1
    label_column: int = 2
    undirected: bool = True
    bigger_ppi_path: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSplitConfig:
    """Immutable options controlling edge partition loading or generation."""
    index_path: str
    validation_size: float = 0.2
    test_size: float = 0.0
    regenerate: bool = False
    mode: SplitMode = 'random'
