from src.train.pipeline import PipelineStep


class LoadProteinSequenceDataStep(PipelineStep):
    def process(self, context):
        context.ppi_data.get_protein_aac(context.args.pseq_path)
        return context
