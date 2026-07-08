import os
import time

from src.train.pipeline import PipelineStep


class PrepareTrainArtifactsStep(PipelineStep):
    def process(self, context):
        args = context.args

        os.makedirs(args.save_path, exist_ok=True)

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        context.save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
        context.result_file_path = os.path.join(context.save_path, "valid_results.txt")
        config_path = os.path.join(context.save_path, "config.txt")
        os.makedirs(context.save_path, exist_ok=True)

        with open(config_path, 'w') as f:
            args_dict = args.__dict__
            for key in args_dict:
                f.write("{} = {}".format(key, args_dict[key]))
                f.write('\n')
            f.write('\n')
            f.write("train gnn, train_num: {}, valid_num: {}".format(
                len(context.graph.train_mask),
                len(context.graph.val_mask),
            ))
        return context
