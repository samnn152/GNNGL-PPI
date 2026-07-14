"""Load protein sequences into the current PPI dataset."""

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class LoadProteinSequenceDataStep(PipelineStep):
    """Load amino-acid sequences for proteins in the initialized dataset."""

    def process(self, context: TrainContext) -> TrainContext:
        """Read sequence data from the configured protein-sequence path."""
        context.require_ppi_data().get_protein_aac(context.args.pseq_path)
        return context
