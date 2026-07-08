import json
import math
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.common.utils import Metrictor_PPI
from src.datasets.gnn_data import GNN_DATA
from src.graph.subgraph import SubgraphExtractor, SubgraphsData
from src.models.gnn_model import GNNGL_PPI


class ModelEvaluator:
    @staticmethod
    def test(model, graph, test_mask, device):
        valid_pre_result_list = []
        valid_label_list = []
        model.eval()
        batch_size = 10
        valid_steps = math.ceil(len(test_mask) / batch_size)

        for step in tqdm(range(valid_steps)):
            if step == valid_steps - 1:
                valid_edge_id = test_mask[step * batch_size:]
            else:
                valid_edge_id = test_mask[step * batch_size: step * batch_size + batch_size]

            output = model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
            label = graph.edge_attr_1[valid_edge_id]
            label = label.type(torch.FloatTensor).to(device)
            pre_result = (nn.Sigmoid()(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(label.cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
        metrics.show_result()
        print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))
        return valid_pre_result_list

    @classmethod
    def run(cls, args):
        cls._validate_paths(args)
        ppi_data = cls._load_ppi_data(args)
        graph = cls._build_graph(ppi_data)
        ppi_list = cls._edge_list(graph)
        cls._attach_masks(graph, ppi_list, args.index_path)
        model, device = cls._load_model(graph, args.gnn_model, args.fusion_strategy)
        graph.to(device)
        cls._print_results(model, graph, device, args.test_all)

    @staticmethod
    def _validate_paths(args):
        if not os.path.exists(args.index_path):
            raise FileNotFoundError("Index file not found: {}".format(args.index_path))
        if not os.path.exists(args.gnn_model):
            raise FileNotFoundError(
                "Model checkpoint not found: {}. Train a model first or pass --gnn_model.".format(args.gnn_model)
            )

    @staticmethod
    def _load_ppi_data(args):
        ppi_data = GNN_DATA(ppi_path=args.ppi_path)
        ppi_data.get_feature_pretrained(pseq_path=args.pseq_path, pre_emb_path=args.pre_emb_path)
        ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)
        ppi_data.generate_data()
        return ppi_data

    @staticmethod
    def _build_graph(ppi_data):
        graph = SubgraphsData(**ppi_data.data.to_dict())
        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = SubgraphExtractor.extract_subgraphs(
            graph.edge_index,
            graph.x.shape[0],
            num_hops=1,
            walk_length=0,
            p=1,
            q=1,
            repeat=5,
        )
        subgraphs_nodes, subgraphs_edges, hop_indicator = SubgraphExtractor.to_sparse(
            subgraphs_nodes_mask,
            subgraphs_edges_mask,
            hop_indicator_dense,
        )
        graph.combined_subgraphs = SubgraphExtractor.combine_subgraphs(
            graph.edge_index,
            subgraphs_nodes,
            subgraphs_edges,
            num_selected=graph.num_nodes,
            num_nodes=graph.num_nodes,
        )
        graph.subgraphs_batch = subgraphs_nodes[0]
        graph.subgraphs_nodes_mapper = subgraphs_nodes[1]
        graph.subgraphs_edges_mapper = subgraphs_edges[1]
        graph.hop_indicator = hop_indicator
        graph.__num_nodes__ = graph.num_nodes
        return graph

    @staticmethod
    def _edge_list(graph):
        return [list(edge) for edge in graph.edge_index.transpose(0, 1).numpy()]

    @classmethod
    def _attach_masks(cls, graph, ppi_list, index_path):
        with open(index_path, 'r') as f:
            index_dict = json.load(f)

        graph.train_mask = index_dict['train_index']
        graph.val_mask = index_dict['valid_index']
        print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

        node_vision_dict = cls._node_vision_dict(graph, ppi_list)
        cls._print_vision_counts(node_vision_dict)
        graph.test1_mask, graph.test2_mask, graph.test3_mask = cls._split_test_masks(graph, ppi_list, node_vision_dict)
        print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(
            len(graph.test1_mask),
            len(graph.test2_mask),
            len(graph.test3_mask),
        ))

    @staticmethod
    def _node_vision_dict(graph, ppi_list):
        node_vision_dict = {}
        for index in graph.train_mask:
            ppi = ppi_list[index]
            if ppi[0] not in node_vision_dict.keys():
                node_vision_dict[ppi[0]] = 1
            if ppi[1] not in node_vision_dict.keys():
                node_vision_dict[ppi[1]] = 1

        for index in graph.val_mask:
            ppi = ppi_list[index]
            if ppi[0] not in node_vision_dict.keys():
                node_vision_dict[ppi[0]] = 0
            if ppi[1] not in node_vision_dict.keys():
                node_vision_dict[ppi[1]] = 0
        return node_vision_dict

    @staticmethod
    def _print_vision_counts(node_vision_dict):
        vision_num = 0
        unvision_num = 0
        for node in node_vision_dict:
            if node_vision_dict[node] == 1:
                vision_num += 1
            elif node_vision_dict[node] == 0:
                unvision_num += 1
        print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    @staticmethod
    def _split_test_masks(graph, ppi_list, node_vision_dict):
        test1_mask = []
        test2_mask = []
        test3_mask = []

        for index in graph.val_mask:
            ppi = ppi_list[index]
            temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
            if temp == 2:
                test1_mask.append(index)
            elif temp == 1:
                test2_mask.append(index)
            elif temp == 0:
                test3_mask.append(index)
        return test1_mask, test2_mask, test3_mask

    @staticmethod
    def _load_model(graph, model_path, fusion_strategy):
        device = torch.device('cpu')
        model = GNNGL_PPI(
            graph,
            gin_in_feature=256,
            num_layers=1,
            hidden=512,
            use_jk=False,
            train_eps=True,
            feature_fusion=None,
            class_num=7,
            fusion_strategy=fusion_strategy,
        ).to(device)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            if fusion_strategy != 'scalar':
                raise
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print(
                "Loaded legacy scalar checkpoint with missing keys: {}, unexpected keys: {}".format(
                    missing_keys,
                    unexpected_keys,
                )
            )
        return model, device

    @classmethod
    def _print_results(cls, model, graph, device, test_all):
        if test_all:
            print("---------------- valid-test-all result --------------------")
            cls.test(model, graph, graph.val_mask, device)
            return

        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            cls.test(model, graph, graph.test1_mask, device)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            cls.test(model, graph, graph.test2_mask, device)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            cls.test(model, graph, graph.test3_mask, device)
