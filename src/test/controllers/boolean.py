"""Boolean CLI argument helpers for evaluation controllers."""


class EvaluationBooleanArgument:
    @staticmethod
    def parse(s: str) -> bool:
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
