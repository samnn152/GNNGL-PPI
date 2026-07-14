"""Composable model workflow primitives for training setup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from src.train.models.context import TrainContext


class PipelineComponent(ABC):
    """Common interface for one executable unit in a training pipeline."""

    @abstractmethod
    def run(self, context: TrainContext) -> TrainContext:
        """Execute the component and return the context for the next component."""
        raise NotImplementedError


class PipelineStep(PipelineComponent):
    """Base class for a single synchronous context-transformation step.

    Subclasses implement ``process`` only. ``run`` supplies the common
    ``PipelineComponent`` interface used by sequential and parallel pipelines.
    """

    def run(self, context: TrainContext) -> TrainContext:
        """Delegate execution to the concrete step implementation."""
        return self.process(context)

    @abstractmethod
    def process(self, context: TrainContext) -> TrainContext:
        """Read or update the context and return it for the following step."""
        raise NotImplementedError


class TrainingSetupPipeline(PipelineComponent):
    """Run the ordered steps that prepare a training context.

    Each step receives the context returned by the previous step, so their
    order defines the setup dependencies. The completed context is consumed by
    ``main`` to create the training session, options, and observer before the
    model's epoch loop begins.
    """

    def __init__(self, steps: Iterable[PipelineComponent] | None = None) -> None:
        """Initialize the pipeline with an optional ordered collection of steps."""
        self.steps = list(steps or [])

    def add(self, step: PipelineComponent) -> TrainingSetupPipeline:
        """Append a setup step and return this pipeline for chained assembly."""
        self.steps.append(step)
        return self

    def run(self, context: TrainContext) -> TrainContext:
        """Run every step in order and return the fully prepared context."""
        for step in self.steps:
            context = step.run(context)
        return context


class ParallelPipeline(PipelineComponent):
    """Run independent components concurrently against a shared context.

    Components must update disjoint context state or otherwise be thread-safe.
    Unlike ``TrainingSetupPipeline``, this class does not pass one component's
    returned context into another component.
    """

    def __init__(
        self,
        steps: Iterable[PipelineComponent] | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Initialize the component collection and optional worker limit."""
        self.steps = list(steps or [])
        self.max_workers = max_workers

    def add(self, step: PipelineComponent) -> ParallelPipeline:
        """Append a concurrent component and return this pipeline."""
        self.steps.append(step)
        return self

    def run(self, context: TrainContext) -> TrainContext:
        """Run all components concurrently and wait for every one to finish."""
        if not self.steps:
            return context

        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.steps)) as executor:
            futures = [executor.submit(step.run, context) for step in self.steps]
            for future in futures:
                future.result()
        return context
