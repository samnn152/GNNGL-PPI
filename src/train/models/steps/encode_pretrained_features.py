"""Attach pretrained MASSA protein features to the setup context."""

import numpy as np

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class EncodeMASSAProteinFeatureStep(PipelineStep):
    """Load MASSA embeddings and map them to proteins in the PPI dataset."""

    def process(self, context: TrainContext) -> TrainContext:
        """Populate the model-ready protein feature dictionary in place."""
        print("+++++++++++++++++++++++use_get_feature_pretrained++++++++++++++++++++++++++++++++")
        ppi_data = context.require_ppi_data()
        ppi_data.pretrained_emb_init(context.args.pre_emb_path)
        for name in ppi_data.protein_name.keys():
            key = name.split('.', 1)[1] if '.' in name else name
            ppi_data.protein_dict[name] = np.array(ppi_data.pretrained_emb_dict[key])
        print('self.protein_dict', len(ppi_data.protein_dict))
        return context
