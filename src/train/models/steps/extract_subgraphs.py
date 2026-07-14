"""Prepare optional bounded local subgraphs for model input."""

from src.common.data.graph.subgraph import SubgraphExtractor, SubgraphsData
from src.common.models.configuration.graph import SparseSubgraphConfig
from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class ExtractKHopSubgraphStep(PipelineStep):
    """Prepare the global graph and optional bounded k-hop local subgraphs."""

    def process(self, context: TrainContext) -> TrainContext:
        """Attach graph inputs required by the configured local encoder."""
        ppi_data = context.require_ppi_data()
        graph = ppi_data.data
        print(graph)
        graph = SubgraphsData(**graph.to_dict())
        assert graph.x is not None
        node_count = graph.x.shape[0]

        if context.args.feature_source == 'global' or context.args.local_encoder == 'sparse':
            context.graph = graph
            context.ppi_list = ppi_data.ppi_list
            return context

        subgraphs_nodes, subgraphs_edges, hop_indicator = SubgraphExtractor.extract_subgraphs_sparse(
            graph.edge_index,
            node_count,
            SparseSubgraphConfig(
                hops=context.args.subgraph_hops,
                max_nodes=context.args.max_subgraph_nodes,
                max_edges=context.args.max_subgraph_edges,
            ),
        )

        combined_subgraphs = SubgraphExtractor.combine_subgraphs(
            graph.edge_index,
            subgraphs_nodes,
            subgraphs_edges,
            num_nodes=node_count,
        )
        graph.subgraphs_batch = subgraphs_nodes[0]
        graph.subgraphs_nodes_mapper = subgraphs_nodes[1]
        graph.subgraphs_edges_mapper = subgraphs_edges[1]
        graph.combined_subgraphs = combined_subgraphs
        graph.hop_indicator = hop_indicator
        graph.num_nodes = node_count

        context.graph = graph
        context.ppi_list = ppi_data.ppi_list
        return context
