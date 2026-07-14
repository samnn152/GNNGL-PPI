from src.test.controllers.evaluation_controller import EvaluationCLI


class MainTestCLI:
    @staticmethod
    def run() -> None:
        EvaluationCLI.run()


if __name__ == "__main__":
    MainTestCLI.run()
