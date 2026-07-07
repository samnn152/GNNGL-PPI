class PipelineStep:
    def __init__(self):
        self._next_step = None

    def set_next(self, next_step):
        self._next_step = next_step
        return next_step

    def handle(self, context):
        context = self.process(context)
        if self._next_step is None:
            return context
        return self._next_step.handle(context)

    def process(self, context):
        raise NotImplementedError
