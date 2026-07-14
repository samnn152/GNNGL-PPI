"""Mutable model state for the training setup workflow."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.data.datasets.gnn_data import GNN_DATA
from src.common.models.ppi_graph import PPIGraph
from src.train.models.configuration import TrainConfig
from src.train.models.components import TrainingComponents


@dataclass
class TrainContext:
    """Mutable state passed through the ordered training setup pipeline.

    The context starts with resolved arguments only. Setup steps progressively
    attach dataset, graph, model components, and output paths. ``require_*``
    methods make those ordering dependencies explicit at runtime.
    """

    args: TrainConfig
    ppi_data: GNN_DATA | None = None
    graph: PPIGraph | None = None
    ppi_list: list[list[int]] | None = None
    training_components: TrainingComponents | None = None
    save_path: str | None = None
    result_file_path: str | None = None

    def require_ppi_data(self) -> GNN_DATA:
        """Return the initialized PPI dataset or report an invalid step order."""
        if self.ppi_data is None:
            raise RuntimeError("PPI data is not initialized; check the pipeline step order")
        return self.ppi_data

    def require_graph(self) -> PPIGraph:
        """Return the prepared graph or report an invalid step order."""
        if self.graph is None:
            raise RuntimeError("Graph is not initialized; check the pipeline step order")
        return self.graph

    def require_ppi_list(self) -> list[list[int]]:
        """Return PPI pairs after graph extraction has attached them."""
        if self.ppi_list is None:
            raise RuntimeError("PPI list is not initialized; check the pipeline step order")
        return self.ppi_list

    def require_training_components(self) -> TrainingComponents:
        """Return model-side dependencies after component setup has run."""
        if self.training_components is None:
            raise RuntimeError("Training components are not initialized; check the pipeline step order")
        return self.training_components

    def attach_training_components(self, components: TrainingComponents) -> None:
        """Attach the model-side dependencies produced during setup."""
        self.training_components = components
