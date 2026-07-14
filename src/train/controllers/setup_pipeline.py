"""Controller-side composition of the training preparation pipeline."""

from src.train.models.pipeline import TrainingSetupPipeline
from src.train.models.steps import (
    AttachTrainTestEdgeMasksStep,
    BuildPPIGraphStep,
    BuildSequenceSupportFeatureStep,
    BuildTrainingComponentsStep,
    EncodeMASSAProteinFeatureStep,
    ExtractKHopSubgraphStep,
    LoadPPINetworkInputStep,
    LoadProteinSequenceDataStep,
    PartitionDatasetStep,
    PrepareTrainArtifactsStep,
)


class TrainingSetupPipelineFactory:
    """Define the standard setup pipeline used by the training entry point.

    Keeping the ordered composition here makes ``main`` independent of
    individual setup steps and provides one place to inspect or change the
    dependencies required before training starts.
    """

    @staticmethod
    def build() -> TrainingSetupPipeline:
        """Build a fresh pipeline containing the default setup steps in order."""
        return TrainingSetupPipeline([
            LoadPPINetworkInputStep(),
            LoadProteinSequenceDataStep(),
            EncodeMASSAProteinFeatureStep(),
            BuildSequenceSupportFeatureStep(),
            BuildPPIGraphStep(),
            PartitionDatasetStep(),
            ExtractKHopSubgraphStep(),
            AttachTrainTestEdgeMasksStep(),
            BuildTrainingComponentsStep(),
            PrepareTrainArtifactsStep(),
        ])
