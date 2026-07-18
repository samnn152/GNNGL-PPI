"""Prepare filesystem locations and metadata for a training run."""

import os
import time

from src.train.models.pipeline import PipelineStep
from src.train.models.context import TrainContext


class PrepareTrainArtifactsStep(PipelineStep):
    """Create the run directory and write its reproducibility metadata."""

    def process(self, context: TrainContext) -> TrainContext:
        """Attach result paths and save the resolved configuration and split sizes."""
        args = context.args
        graph = context.require_graph()

        os.makedirs(args.save_path, exist_ok=True)

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        context.save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
        context.result_file_path = os.path.join(context.save_path, "valid_results.txt")
        config_path = os.path.join(context.save_path, "config.txt")
        os.makedirs(context.save_path, exist_ok=True)

        with open(config_path, 'w') as f:
            args_dict = args.as_dict()
            for key in args_dict:
                f.write("{} = {}".format(key, args_dict[key]))
                f.write('\n')
            f.write('\n')
            f.write("train gnn, train_num: {}, valid_num: {}, test_num: {}".format(
                len(graph.train_mask),
                len(graph.val_mask),
                len(graph.test_mask),
            ))
        return context
