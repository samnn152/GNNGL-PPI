"""Boolean CLI argument helpers for evaluation controllers."""


class EvaluationBooleanArgument:
    """Parse textual boolean values accepted by the evaluation CLI."""

    @staticmethod
    def parse(s: str) -> bool:
        """Convert ``True`` or ``False`` text and reject all other values."""
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
