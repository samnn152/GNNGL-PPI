import torch

from src.train.pipeline import PipelineStep


class AttachTrainTestEdgeMasksStep(PipelineStep):
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
