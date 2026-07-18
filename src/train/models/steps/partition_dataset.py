"""Partition PPI edges for training and validation."""

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext
from src.common.models.configuration.data import DatasetSplitConfig


class PartitionDatasetStep(PipelineStep):
    """Load or generate the training and validation edge partitions."""

    def process(self, context: TrainContext) -> TrainContext:
        """Apply the configured split mode and persist split indices in the dataset."""
        print("----------------------- start split train and valid index -------------------")
        print("whether to split new train and valid index file, {}".format(context.args.split_new))
        if context.args.split_new:
            print("use {} method to split".format(context.args.split_mode))
        context.require_ppi_data().split_dataset(DatasetSplitConfig(
            index_path=context.args.train_valid_index_path,
            validation_size=context.args.validation_size,
            test_size=context.args.test_size,
            regenerate=context.args.split_new,
            mode=context.args.split_mode,
        ))
        print("----------------------- Done split train and valid index -------------------")
        return context
