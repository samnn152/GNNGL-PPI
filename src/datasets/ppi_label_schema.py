class PPILabelSchema:
    CLASS_TO_INDEX = {
        'reaction': 0,
        'binding': 1,
        'ptmod': 2,
        'activation': 3,
        'inhibition': 4,
        'catalysis': 5,
        'expression': 6,
    }
    CLASS_COUNT = 7

    @classmethod
    def empty_label(cls):
        return [0] * cls.CLASS_COUNT
