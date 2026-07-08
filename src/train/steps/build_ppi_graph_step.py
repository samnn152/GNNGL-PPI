from src.train.pipeline import PipelineStep


class BuildPPIGraphStep(PipelineStep):
    def process(self, context):
        context.ppi_data.generate_data()
        return context
