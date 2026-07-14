# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Checkpoint evaluation model."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Mapping, cast

import torch
from torch import Tensor
from tqdm import tqdm

from src.common.utils import Metrictor_PPI
from src.common.device import DeviceResolver
from src.common.models.configuration.data import DatasetSplit, GNNDataConfig
from src.common.data.datasets.gnn_data import GNN_DATA
from src.common.models.configuration.graph import SparseSubgraphConfig
from src.common.models.ppi_graph import PPIGraph
from src.common.data.graph.subgraph import SubgraphExtractor, SubgraphsData
from src.common.models.configuration.gnn import GNNModelConfig
from src.common.models.core.gnngl import GNNGL_PPI
from src.test.models.configuration import EvaluationConfig


@dataclass(frozen=True, slots=True)
class EvaluationSession:
    model: GNNGL_PPI
    graph: PPIGraph
    device: torch.device
    batch_size: int


@dataclass(frozen=True, slots=True)
class ModelCheckpointRequest:
    path: str
    model: GNNModelConfig
    device: torch.device


class ModelEvaluator:
    @staticmethod
    def _coerce_graph_float32(graph: PPIGraph) -> None:
        for key, value in graph.to_dict().items():
            if isinstance(value, Tensor) and value.dtype == torch.float64:
                setattr(graph, key, value.float())

    @staticmethod
    def test(session: EvaluationSession, test_mask: list[int]) -> Metrictor_PPI:
        predictions: list[Tensor] = []
        labels: list[Tensor] = []
        session.model.eval()
        step_count = math.ceil(len(test_mask) / session.batch_size)
        if step_count == 0:
            raise ValueError("Evaluation split contains no edges")
        with torch.no_grad():
            node_embeddings = session.model.encode_nodes(
                session.graph.x, session.graph.edge_index, session.graph
            )
            for step in tqdm(range(step_count)):
                edge_ids = test_mask[step * session.batch_size:(step + 1) * session.batch_size]
                output = session.model.decode_edges(node_embeddings, session.graph.edge_index, edge_ids)
                label = session.graph.edge_attr_1[edge_ids].to(dtype=torch.float32, device=session.device)
                prediction = output.sigmoid().gt(0.5).to(dtype=torch.float32, device=session.device)
                predictions.append(prediction.cpu())
                labels.append(label.cpu())
        metrics = Metrictor_PPI(torch.cat(predictions), torch.cat(labels))
        print(f"Recall: {metrics.Recall}, Precision: {metrics.Precision}, F1: {metrics.F1}")
        return metrics

    @classmethod
    def run(cls, config: EvaluationConfig) -> dict[str, Metrictor_PPI]:
        cls._validate_paths(config)
        ppi_data = cls._load_ppi_data(config)
        graph = cls._build_graph(ppi_data, config)
        ppi_list = cls._edge_list(graph)
        cls._attach_masks(graph, ppi_list, config.index_path)
        device = DeviceResolver.resolve(config.device)
        cls._coerce_graph_float32(graph)
        model_config = GNNModelConfig(
            fusion_strategy=config.fusion_strategy,
            feature_source=config.feature_source,
            local_encoder=config.local_encoder,
        )
        model = cls._load_model(ModelCheckpointRequest(config.gnn_model, model_config, device))
        if hasattr(graph, 'x_origin'):
            del graph.x_origin
        graph.to(device)
        session = EvaluationSession(model, graph, device, config.test_batch_size)
        return cls._collect_results(session, config.test_all)

    @staticmethod
    def _validate_paths(config: EvaluationConfig) -> None:
        if not os.path.exists(config.index_path):
            raise FileNotFoundError(f"Index file not found: {config.index_path}")
        if not os.path.exists(config.gnn_model):
            raise FileNotFoundError(
                f"Model checkpoint not found: {config.gnn_model}. Train a model first or pass --gnn_model."
            )

    @staticmethod
    def _load_ppi_data(config: EvaluationConfig) -> GNN_DATA:
        ppi_data = GNN_DATA(GNNDataConfig(ppi_path=config.ppi_path))
        ppi_data.get_feature_pretrained(config.pseq_path, config.pre_emb_path)
        ppi_data.get_feature_origin(config.pseq_path, config.vec_path)
        ppi_data.generate_data()
        return ppi_data

    @staticmethod
    def _build_graph(ppi_data: GNN_DATA, config: EvaluationConfig) -> PPIGraph:
        graph = SubgraphsData(**ppi_data.data.to_dict())
        if config.feature_source == 'global' or config.local_encoder == 'sparse':
            return graph
        node_count = graph.x.shape[0]
        nodes, edges, hops = SubgraphExtractor.extract_subgraphs_sparse(
            graph.edge_index,
            node_count,
            SparseSubgraphConfig(
                hops=config.subgraph_hops,
                max_nodes=config.max_subgraph_nodes,
                max_edges=config.max_subgraph_edges,
            ),
        )
        graph.combined_subgraphs = SubgraphExtractor.combine_subgraphs(
            graph.edge_index, nodes, edges, num_nodes=node_count
        )
        graph.subgraphs_batch = nodes[0]
        graph.subgraphs_nodes_mapper = nodes[1]
        graph.subgraphs_edges_mapper = edges[1]
        graph.hop_indicator = hops
        graph.num_nodes = node_count
        return graph

    @staticmethod
    def _edge_list(graph: PPIGraph) -> list[list[int]]:
        return cast(list[list[int]], graph.edge_index.transpose(0, 1).cpu().tolist())

    @classmethod
    def _attach_masks(cls, graph: PPIGraph, ppi_list: list[list[int]], index_path: str) -> None:
        with open(index_path, 'r') as file:
            index_dict = cast(DatasetSplit, json.load(file))
        graph.train_mask = index_dict['train_index']
        graph.val_mask = index_dict['valid_index']
        node_visibility = cls._node_visibility(graph, ppi_list)
        cls._print_visibility_counts(node_visibility)
        graph.test1_mask, graph.test2_mask, graph.test3_mask = cls._split_test_masks(
            graph, ppi_list, node_visibility
        )

    @staticmethod
    def _node_visibility(graph: PPIGraph, ppi_list: list[list[int]]) -> dict[int, int]:
        visibility: dict[int, int] = {}
        for index in graph.train_mask:
            for node in ppi_list[index]:
                visibility[node] = 1
        for index in graph.val_mask:
            for node in ppi_list[index]:
                visibility.setdefault(node, 0)
        return visibility

    @staticmethod
    def _print_visibility_counts(node_visibility: Mapping[int, int]) -> None:
        seen_count = sum(value == 1 for value in node_visibility.values())
        unseen_count = sum(value == 0 for value in node_visibility.values())
        print(f"vision node num: {seen_count}, unvision node num: {unseen_count}")

    @staticmethod
    def _split_test_masks(
        graph: PPIGraph,
        ppi_list: list[list[int]],
        node_visibility: Mapping[int, int],
    ) -> tuple[list[int], list[int], list[int]]:
        masks: tuple[list[int], list[int], list[int]] = ([], [], [])
        for index in graph.val_mask:
            source, target = ppi_list[index]
            visibility = node_visibility[source] + node_visibility[target]
            masks[2 - visibility].append(index)
        return masks

    @staticmethod
    def _load_model(request: ModelCheckpointRequest) -> GNNGL_PPI:
        model = GNNGL_PPI(request.model).to(request.device)
        loaded = torch.load(request.path, map_location=request.device)
        checkpoint = cast(dict[str, Tensor], cast(dict[str, object], loaded)['state_dict'])
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            if request.model.fusion_strategy != 'scalar':
                raise
            incompatible = model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded legacy scalar checkpoint: {incompatible}")
        return model

    @classmethod
    def _collect_results(
        cls,
        session: EvaluationSession,
        test_all: bool,
    ) -> dict[str, Metrictor_PPI]:
        if test_all:
            return {'test_all': cls.test(session, session.graph.val_mask)}
        results: dict[str, Metrictor_PPI] = {}
        for name, mask in (
            ('test1', session.graph.test1_mask),
            ('test2', session.graph.test2_mask),
            ('test3', session.graph.test3_mask),
        ):
            if mask:
                results[name] = cls.test(session, mask)
        return results
