class EvaluationBooleanArgument:
    @staticmethod
    def parse(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
