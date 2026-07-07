import os
import time

import torch
import torch.nn as nn

from models.asl import AsymmetricLossOptimized
from data_processing.gnn_data import GNN_DATA
from models.gnn_model import GNNGL_PPI
from graph.subgraph import SubgraphsData, combine_subgraphs, extract_subgraphs, to_sparse
from .pipeline import PipelineStep


class LoadPPIDataStep(PipelineStep):
    def process(self, context):
        context.ppi_data = GNN_DATA(ppi_path=context.args.ppi_path)
        return context


class BuildProteinFeaturesStep(PipelineStep):
    def process(self, context):
        print("+++++++++++++++++++++++use_get_feature_pretrained++++++++++++++++++++++++++++++++")
        context.ppi_data.get_feature_pretrained(
            pseq_path=context.args.pseq_path,
            pre_emb_path=context.args.pre_emb_path,
        )
        print("+++++++++++++++++++++++use_get_feature_vec+++++++++++++++++++++++++++++++++")
        context.ppi_data.get_feature_origin(
            pseq_path=context.args.pseq_path,
            vec_path=context.args.vec_path,
        )
        return context


class BuildGraphDataStep(PipelineStep):
    def process(self, context):
        context.ppi_data.generate_data()
        return context


class SplitDatasetStep(PipelineStep):
    def process(self, context):
        print("----------------------- start split train and valid index -------------------")
        print("whether to split new train and valid index file, {}".format(context.args.split_new))
        if context.args.split_new:
            print("use {} method to split".format(context.args.split_mode))
        context.ppi_data.split_dataset(
            context.args.train_valid_index_path,
            random_new=context.args.split_new,
            mode=context.args.split_mode,
        )
        print("----------------------- Done split train and valid index -------------------")
        return context


class ExtractSubgraphsStep(PipelineStep):
    def process(self, context):
        graph = context.ppi_data.data
        print(graph)
        graph = SubgraphsData(**graph.to_dict())
        assert graph.x is not None

        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(
            graph.edge_index,
            graph.x.shape[0],
            num_hops=1,
            walk_length=0,
            p=1,
            q=1,
            repeat=5,
        )
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
            subgraphs_nodes_mask,
            subgraphs_edges_mask,
            hop_indicator_dense,
        )

        combined_subgraphs = combine_subgraphs(
            graph.edge_index,
            subgraphs_nodes,
            subgraphs_edges,
            num_selected=graph.num_nodes,
            num_nodes=graph.num_nodes,
        )
        graph.subgraphs_batch = subgraphs_nodes[0]
        graph.subgraphs_nodes_mapper = subgraphs_nodes[1]
        graph.subgraphs_edges_mapper = subgraphs_edges[1]
        graph.combined_subgraphs = combined_subgraphs
        graph.hop_indicator = hop_indicator
        graph.__num_nodes__ = graph.num_nodes

        context.graph = graph
        context.ppi_list = context.ppi_data.ppi_list
        return context


class AttachTrainValidationMasksStep(PipelineStep):
    def process(self, context):
        graph = context.graph
        ppi_data = context.ppi_data

        graph.train_mask = ppi_data.ppi_split_dict['train_index']
        graph.val_mask = ppi_data.ppi_split_dict['valid_index']

        print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

        graph.edge_index_got = torch.cat(
            (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
        graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]),
                                        dim=0)
        graph.edge_mul_type_got = torch.cat((graph.edge_mul[graph.train_mask], graph.edge_mul[graph.train_mask]),
                                            dim=0)
        graph.train_mask_got = [i for i in range(len(graph.train_mask))]
        return context


class BuildTrainingObjectsStep(PipelineStep):
    def process(self, context):
        args = context.args

        context.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(context.device)
        context.graph.to(context.device)

        context.model = GNNGL_PPI(context.graph, gin_in_feature=256, num_layers=1, hidden=512, use_jk=False,
                                  train_eps=True, feature_fusion=None, class_num=7).to(context.device)

        context.optimizer = torch.optim.Adam(context.model.parameters(), lr=0.001, weight_decay=5e-4)

        context.scheduler = None
        if args.use_lr_scheduler:
            context.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                context.optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                verbose=True,
            )

        context.loss_fn = nn.BCEWithLogitsLoss().to(context.device)
        context.loss_asl = AsymmetricLossOptimized(
            gamma_neg=1,
            gamma_pos=0,
            clip=0.05,
            disable_torch_grad_focal_loss=True,
        ).to(context.device)
        return context


class PrepareOutputStep(PipelineStep):
    def process(self, context):
        args = context.args

        os.makedirs(args.save_path, exist_ok=True)

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        context.save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
        context.result_file_path = os.path.join(context.save_path, "valid_results.txt")
        config_path = os.path.join(context.save_path, "config.txt")
        os.makedirs(context.save_path, exist_ok=True)

        with open(config_path, 'w') as f:
            args_dict = args.__dict__
            for key in args_dict:
                f.write("{} = {}".format(key, args_dict[key]))
                f.write('\n')
            f.write('\n')
            f.write("train gnn, train_num: {}, valid_num: {}".format(
                len(context.graph.train_mask),
                len(context.graph.val_mask),
            ))
        return context


def build_training_pipeline():
    first_step = LoadPPIDataStep()
    first_step.set_next(BuildProteinFeaturesStep()) \
        .set_next(BuildGraphDataStep()) \
        .set_next(SplitDatasetStep()) \
        .set_next(ExtractSubgraphsStep()) \
        .set_next(AttachTrainValidationMasksStep()) \
        .set_next(BuildTrainingObjectsStep()) \
        .set_next(PrepareOutputStep())
    return first_step
