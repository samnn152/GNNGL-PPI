"""MVC controller for model evaluation."""

from src.test.models import ModelEvaluator
from src.common.data.reporting import append_test_result
from src.test.controllers.arguments import EvaluationArgumentParser
from src.test.controllers.defaults import EvaluationConfigDefaults


class EvaluationCLI:
    @staticmethod
    def run() -> None:
        args = EvaluationConfigDefaults.apply(EvaluationArgumentParser.build().parse_args())
        results = ModelEvaluator.run(args)
        if args.metrics_csv:
            for evaluation_split, metrics in results.items():
                append_test_result(args.metrics_csv, args, evaluation_split, metrics)


if __name__ == "__main__":
    EvaluationCLI.run()
