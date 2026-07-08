from src.graph.subgraph import SubgraphExtractor, SubgraphsData
from src.train.pipeline import PipelineStep


class ExtractKHopSubgraphStep(PipelineStep):
    def process(self, context):
        graph = context.ppi_data.data
        print(graph)
        graph = SubgraphsData(**graph.to_dict())
        assert graph.x is not None

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

        combined_subgraphs = SubgraphExtractor.combine_subgraphs(
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
