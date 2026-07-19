"""Concrete file-backed PPI dataset preparation facade."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from src.common.utils import UnionFindSet
from src.common.models.configuration.data import DatasetSplit, DatasetSplitConfig, GNNDataConfig
from src.common.data.datasets.splitting.splitter import DatasetSplitter
from src.common.data.datasets.builders.ppi_graph import PPIGraphBuilder
from src.common.data.datasets.parsing.ppi_network import PPINetworkParser
from src.common.data.datasets.features.pretrained import PretrainedProteinFeatureEncoder
from src.common.data.datasets.parsing.protein_sequences import ProteinSequenceStore
from src.common.data.datasets.features.sequence_vectors import SequenceVectorEncoder
from src.common.models.ppi_graph import PPIGraph

FloatArray = NDArray[np.floating[Any]]


class GNN_DATA:
    """All dataset state, initialized from one typed configuration field."""

    def __init__(self, config: GNNDataConfig) -> None:
        """Parse the configured PPI network and initialize empty derived state."""
        self.config = config
        self.ppi_path = config.ppi_path
        self.max_len = config.max_sequence_length
        self.protein_dict: dict[str, FloatArray] = {}
        self.protein_dict_origin: dict[str, FloatArray] = {}
        self.pro_go_edge_type: list[int] = []
        self.pro_go_edge: list[list[int]] = []

        parsed_network = PPINetworkParser(config).parse()
        self.ppi_list = parsed_network['ppi_list']
        self.origin_ppi_list = parsed_network['origin_ppi_list']
        self.ppi_dict = parsed_network['ppi_dict']
        self.ppi_label_list = parsed_network['ppi_label_list']
        self.protein_name = parsed_network['protein_name']
        self.node_num = parsed_network['node_num']
        self.edge_num = parsed_network['edge_num']

        self.pseq_path: str | None = None
        self.pseq_dict: dict[str, str] = {}
        self.protein_len: list[int] = []
        self.acid2vec: dict[str, FloatArray] = {}
        self.dim = 0
        self.pvec_dict: dict[str, FloatArray] = {}
        self.pretrained_emb_dict: dict[str, FloatArray] = {}
        self.ppi_split_dict: DatasetSplit = {
            'train_index': [],
            'valid_index': [],
            'test_index': [],
        }
        self.data: PPIGraph
        self.ufs: UnionFindSet
        self.edge_index: Tensor
        self.edge_attr: Tensor
        self.mul_type: Tensor
        self.x: Tensor
        self.x_origin: Tensor

    def get_protein_aac(self, pseq_path: str) -> None:
        """Load protein sequences and their lengths from disk."""
        self.pseq_path = pseq_path
        self.pseq_dict, self.protein_len = ProteinSequenceStore().load(pseq_path)

    def embed_normal(self, sequence: FloatArray, dim: int) -> FloatArray:
        """Pad or trim one encoded protein sequence to the configured length."""
        return SequenceVectorEncoder(max_len=self.max_len).pad_or_trim(sequence, dim)

    def vectorize(self, vec_path: str) -> None:
        """Encode loaded sequences with amino-acid support vectors."""
        self.acid2vec, self.dim, self.pvec_dict = SequenceVectorEncoder(max_len=self.max_len).vectorize(
            self.pseq_dict, vec_path
        )

    def _pretrained_key(self, protein_name: str) -> str:
        """Return the embedding lookup key for a protein identifier."""
        return PretrainedProteinFeatureEncoder.pretrained_key(protein_name)

    def _sequence_embedding(self, sequence: str, dim: int = 512) -> FloatArray:
        """Return a deterministic fallback embedding for one sequence."""
        return PretrainedProteinFeatureEncoder.sequence_embedding(sequence, dim=dim)

    def pretrained_emb_init(self, pre_emb_path: str) -> None:
        """Load pretrained embeddings or initialize fallback embeddings."""
        self.pretrained_emb_dict = PretrainedProteinFeatureEncoder.load_or_fallback(pre_emb_path, self.pseq_dict)

    def get_feature_pretrained(self, pseq_path: str, pre_emb_path: str) -> None:
        """Attach pretrained or fallback vectors for every graph protein."""
        self.get_protein_aac(pseq_path)
        self.pretrained_emb_init(pre_emb_path)
        self.protein_dict = PretrainedProteinFeatureEncoder.build_protein_features(
            self.protein_name, self.pretrained_emb_dict
        )
        print(f"pretrained protein feature num: {len(self.protein_dict)}")

    def get_feature_origin(self, pseq_path: str, vec_path: str) -> None:
        """Attach original amino-acid matrix features for every graph protein."""
        self.get_protein_aac(pseq_path)
        self.vectorize(vec_path)
        self.protein_dict_origin = {
            protein_name: self.pvec_dict[protein_name]
            for protein_name in tqdm(self.protein_name.keys())
        }

    def get_connected_num(self) -> None:
        """Calculate and retain the graph's connected-component structure."""
        self.ufs = PPIGraphBuilder.connected_components_count(self.node_num, self.ppi_list)

    def generate_data(self) -> None:
        """Build the tensor graph from parsed interactions and encoded features."""
        graph_data = PPIGraphBuilder.build(
            node_num=self.node_num,
            protein_names=self.protein_name,
            ppi_list=self.ppi_list,
            ppi_label_list=self.ppi_label_list,
            protein_features=self.protein_dict,
            support_features=self.protein_dict_origin,
        )
        self.data, self.ufs, self.edge_index, self.edge_attr, self.mul_type, self.x, self.x_origin = graph_data

    def split_dataset(self, config: DatasetSplitConfig) -> None:
        """Load or create train, validation, and test edge partitions."""
        self.ppi_split_dict = DatasetSplitter.split(self.ppi_list, self.edge_num, config)
