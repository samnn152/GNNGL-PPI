from src.test.controllers.evaluation_controller import EvaluationCLI


class MainTestCLI:
    """Expose model evaluation through the repository's legacy test entry point."""

    @staticmethod
    def run() -> None:
        """Delegate argument parsing and evaluation execution to the controller."""
        EvaluationCLI.run()


if __name__ == "__main__":
    MainTestCLI.run()
