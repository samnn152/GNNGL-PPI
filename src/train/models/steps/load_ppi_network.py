"""Initialize the PPI data source used by training setup."""

from src.common.data.datasets.gnn_data import GNN_DATA
from src.common.models.configuration.data import GNNDataConfig
from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class LoadPPINetworkInputStep(PipelineStep):
    """Initialize the PPI dataset from the configured network input path."""

    def process(self, context: TrainContext) -> TrainContext:
        """Create and attach the mutable dataset used by later setup steps."""
        context.ppi_data = GNN_DATA(GNNDataConfig(ppi_path=context.args.ppi_path))
        return context
