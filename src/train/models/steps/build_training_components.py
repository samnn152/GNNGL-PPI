"""Create and attach model-side dependencies during training setup."""

from __future__ import annotations

from src.train.models.context import TrainContext
from src.train.models.pipeline import PipelineStep
from src.train.models.training_components import TrainingComponentFactory


class BuildTrainingComponentsStep(PipelineStep):
    """Prepare the graph and attach model-side dependencies to the context."""

    def process(self, context: TrainContext) -> TrainContext:
        """Build components, move graph tensors, and attach both for training."""
        graph = context.require_graph()
        components = TrainingComponentFactory().build(context.args)
        graph.prepare_for_training(components.device)
        context.attach_training_components(components)
        return context
