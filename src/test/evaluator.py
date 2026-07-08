from src.test.evaluation_argument_parser import EvaluationArgumentParser
from src.test.evaluation_config_defaults import EvaluationConfigDefaults
from src.test.model_evaluator import ModelEvaluator


class EvaluationCLI:
    @staticmethod
    def run():
        args = EvaluationConfigDefaults.apply(EvaluationArgumentParser.build().parse_args())
        ModelEvaluator.run(args)


if __name__ == "__main__":
    EvaluationCLI.run()
