# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Optional analysis models for trained representations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from lime import lime_tabular
from numpy.typing import NDArray
from torch import Tensor, nn

from src.common.models.ppi_graph import PPIGraph
from src.common.models.configuration.gnn import EdgePredictionBatch
from src.common.models.core.gnngl import GNNGL_PPI
from src.test.models.evaluator import EvaluationSession, ModelEvaluator


@dataclass(frozen=True, slots=True)
class FeatureHookRequest:
    """Describe an intermediate model activation to extract."""
    graph: PPIGraph
    edge_ids: list[int] | Tensor
    layer_name: str


@dataclass(frozen=True, slots=True)
class PermutationImportanceRequest:
    """Configure permutation-importance analysis for an evaluation split."""
    session: EvaluationSession
    test_mask: list[int]
    feature_count: int = 20
    permutations: int = 100


@dataclass(frozen=True, slots=True)
class LimeExplanationRequest:
    """Configure a local LIME explanation for one node and target class."""
    session: EvaluationSession
    edge_ids: list[int] | Tensor
    target_class: int
    node_id: int = 0
    feature_count: int = 2


class EvaluationToolkit:
    """Compute model-analysis data without rendering presentation output."""
    @staticmethod
    def extract_features(model: GNNGL_PPI, request: FeatureHookRequest) -> Tensor:
        """Capture and return one named layer's activation for an edge batch."""
        activations: dict[str, Tensor] = {}

        def hook_fn(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            """Store a detached tensor emitted by the requested layer."""
            if not isinstance(output, Tensor):
                raise TypeError(f"Layer {request.layer_name} did not return a Tensor")
            activations[request.layer_name] = output.detach()

        target_layer = model.get_submodule(request.layer_name)
        hook = target_layer.register_forward_hook(hook_fn)
        try:
            model(EdgePredictionBatch(request.graph, request.graph.edge_index, request.edge_ids))
        finally:
            hook.remove()
        return activations[request.layer_name]

    @staticmethod
    def normalize_correlation(first: Tensor, second: Tensor) -> Tensor:
        """Map a pairwise dot-product matrix linearly into the range [-1, 1]."""
        correlation = torch.matmul(first, second.t())
        span = correlation.max() - correlation.min()
        if float(span.item()) == 0:
            return torch.zeros_like(correlation)
        return 2 * (correlation - correlation.min()) / span - 1

    @staticmethod
    def feature_correlation(feature_map: Tensor) -> NDArray[np.float64]:
        """Convert the first ten feature rows into a correlation matrix for a View."""
        selected = feature_map[:10].detach().cpu()
        return cast(NDArray[np.float64], np.corrcoef(selected.numpy()))

    @classmethod
    def permutation_importance(cls, request: PermutationImportanceRequest) -> NDArray[np.float64]:
        """Measure F1 degradation after independently permuting input features."""
        del cls
        graph = request.session.graph
        baseline = ModelEvaluator.test(request.session, request.test_mask).F1
        original_features = graph.x.clone()
        importances = np.zeros(request.feature_count, dtype=np.float64)
        try:
            for feature_index in range(request.feature_count):
                scores: list[float] = []
                for _ in range(request.permutations):
                    permutation = torch.randperm(graph.x.shape[0], device=graph.x.device)
                    graph.x[:, feature_index] = original_features[permutation, feature_index]
                    scores.append(ModelEvaluator.test(request.session, request.test_mask).F1)
                importances[feature_index] = baseline - float(np.mean(scores))
                graph.x[:, feature_index] = original_features[:, feature_index]
        finally:
            graph.x = original_features
        return importances

    @staticmethod
    def explain_instance(request: LimeExplanationRequest) -> list[tuple[str, float]]:
        """Return LIME feature contributions for one requested prediction target."""
        session = request.session
        graph = session.graph
        original_features = graph.x.clone()
        training_data = original_features.detach().cpu().numpy()
        explainer = lime_tabular.LimeTabularExplainer(training_data, mode='regression')

        def predict(rows: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate LIME perturbations while restoring the original graph features."""
            scores: list[float] = []
            try:
                for row in rows:
                    graph.x[request.node_id] = torch.as_tensor(
                        row, dtype=graph.x.dtype, device=graph.x.device
                    )
                    output = session.model(EdgePredictionBatch(
                        graph, graph.edge_index, request.edge_ids
                    ))
                    scores.append(float(output.sigmoid()[:, request.target_class].mean().item()))
            finally:
                graph.x = original_features.clone()
            return np.asarray(scores, dtype=np.float64)

        explanation = explainer.explain_instance(
            training_data[request.node_id],
            cast(Any, predict),
            num_features=request.feature_count,
        )
        return cast(list[tuple[str, float]], explanation.as_list())
