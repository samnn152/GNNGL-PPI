"""Build graph tensors from prepared PPI features."""

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class BuildPPIGraphStep(PipelineStep):
    """Generate graph tensors from the loaded and encoded PPI dataset."""

    def process(self, context: TrainContext) -> TrainContext:
        """Populate the dataset's graph representation in place."""
        context.require_ppi_data().generate_data()
        return context
