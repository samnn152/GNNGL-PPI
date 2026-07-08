from concurrent.futures import ThreadPoolExecutor


class PipelineComponent:
    def run(self, context):
        raise NotImplementedError


class PipelineStep(PipelineComponent):
    def run(self, context):
        return self.process(context)

    def process(self, context):
        raise NotImplementedError


class TrainPipeline(PipelineComponent):
    def __init__(self, steps=None):
        self.steps = list(steps or [])

    def add(self, step):
        self.steps.append(step)
        return self

    def run(self, context):
        for step in self.steps:
            context = step.run(context)
        return context


class ParallelPipeline(PipelineComponent):
    def __init__(self, steps=None, max_workers=None):
        self.steps = list(steps or [])
        self.max_workers = max_workers

    def add(self, step):
        self.steps.append(step)
        return self

    def run(self, context):
        if not self.steps:
            return context

        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.steps)) as executor:
            futures = [executor.submit(step.run, context) for step in self.steps]
            for future in futures:
                future.result()
        return context
