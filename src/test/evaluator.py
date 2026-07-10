from src.test.evaluation_argument_parser import EvaluationArgumentParser
from src.test.evaluation_config_defaults import EvaluationConfigDefaults
from src.test.model_evaluator import ModelEvaluator
from src.train.experiment_results import append_test_result


class EvaluationCLI:
    @staticmethod
    def run():
        args = EvaluationConfigDefaults.apply(EvaluationArgumentParser.build().parse_args())
        results = ModelEvaluator.run(args)
        if args.metrics_csv:
            for evaluation_split, metrics in results.items():
                append_test_result(args.metrics_csv, args, evaluation_split, metrics)


if __name__ == "__main__":
    EvaluationCLI.run()
