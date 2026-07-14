"""Prepare sequence-derived support features for training."""

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class BuildSequenceSupportFeatureStep(PipelineStep):
    """Load sequence-support vectors and retain their original representation."""

    def process(self, context: TrainContext) -> TrainContext:
        """Vectorize proteins from ``vec_path`` and cache vectors by protein name."""
        print("+++++++++++++++++++++++use_get_feature_vec+++++++++++++++++++++++++++++++++")
        ppi_data = context.require_ppi_data()
        ppi_data.vectorize(context.args.vec_path)
        ppi_data.protein_dict_origin = {}
        for name in ppi_data.protein_name.keys():
            ppi_data.protein_dict_origin[name] = ppi_data.pvec_dict[name]
        return context
