from src.train.pipeline import PipelineStep


class PartitionDatasetStep(PipelineStep):
    def process(self, context):
        print("----------------------- start split train and valid index -------------------")
        print("whether to split new train and valid index file, {}".format(context.args.split_new))
        if context.args.split_new:
            print("use {} method to split".format(context.args.split_mode))
        context.ppi_data.split_dataset(
            context.args.train_valid_index_path,
            random_new=context.args.split_new,
            mode=context.args.split_mode,
        )
        print("----------------------- Done split train and valid index -------------------")
        return context
