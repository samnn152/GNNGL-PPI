from src.train.pipeline import PipelineStep


class BuildSequenceSupportFeatureStep(PipelineStep):
    def process(self, context):
        print("+++++++++++++++++++++++use_get_feature_vec+++++++++++++++++++++++++++++++++")
        context.ppi_data.vectorize(context.args.vec_path)
        context.ppi_data.protein_dict_origin = {}
        for name in context.ppi_data.protein_name.keys():
            context.ppi_data.protein_dict_origin[name] = context.ppi_data.pvec_dict[name]
        return context
