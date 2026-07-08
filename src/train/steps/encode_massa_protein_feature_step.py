import numpy as np

from src.train.pipeline import PipelineStep


class EncodeMASSAProteinFeatureStep(PipelineStep):
    def process(self, context):
        print("+++++++++++++++++++++++use_get_feature_pretrained++++++++++++++++++++++++++++++++")
        context.ppi_data.pretrained_emb_init(context.args.pre_emb_path)
        for name in context.ppi_data.protein_name.keys():
            key = context.ppi_data._pretrained_key(name)
            context.ppi_data.protein_dict[name] = np.array(context.ppi_data.pretrained_emb_dict[key])
        print('self.protein_dict', len(context.ppi_data.protein_dict))
        return context
