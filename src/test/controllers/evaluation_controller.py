"""MVC controller for model evaluation."""

from src.common.models.core.gnngl import GNNGL_PPI
from src.test.models import ModelEvaluator
from src.test.models.toolkit import EvaluationToolkit, FeatureHookRequest
from src.test.views import EvaluationAnalysisView
from src.common.data.reporting import append_test_result
from src.test.controllers.arguments import EvaluationArgumentParser
from src.test.controllers.defaults import EvaluationConfigDefaults


class EvaluationCLI:
    """Coordinate CLI parsing, model evaluation, and result persistence."""

    @staticmethod
    def run() -> None:
        """Execute one evaluation request and append each split's metrics."""
        args = EvaluationConfigDefaults.apply(EvaluationArgumentParser.build().parse_args())
        results = ModelEvaluator.run(args)
        if args.metrics_csv:
            for evaluation_split, metrics in results.items():
                append_test_result(args.metrics_csv, args, evaluation_split, metrics)

    @staticmethod
    def render_feature_correlation(
        model: GNNGL_PPI,
        request: FeatureHookRequest,
        output_path: str | None = None,
    ) -> None:
        """Coordinate feature extraction and correlation-heatmap presentation."""
        feature_map = EvaluationToolkit.extract_features(model, request)
        correlation = EvaluationToolkit.feature_correlation(feature_map)
        EvaluationAnalysisView.plot_correlation(correlation, output_path)


if __name__ == "__main__":
    EvaluationCLI.run()
