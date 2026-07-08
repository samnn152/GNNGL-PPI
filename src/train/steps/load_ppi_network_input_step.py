from src.datasets.gnn_data import GNN_DATA
from src.train.pipeline import PipelineStep


class LoadPPINetworkInputStep(PipelineStep):
    def process(self, context):
        context.ppi_data = GNN_DATA(ppi_path=context.args.ppi_path)
        return context
