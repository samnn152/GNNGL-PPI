import numpy as np
import torch

from training.args import apply_default_config, build_parser
from training.context import TrainingContext
from training.steps import build_training_pipeline
from training.trainer import train
from training.ui import build_training_observer

# from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def require_training_context(context):
    required_fields = [
        "model",
        "graph",
        "ppi_list",
        "loss_fn",
        "loss_asl",
        "optimizer",
        "device",
        "result_file_path",
        "save_path",
    ]
    missing_fields = [field for field in required_fields if getattr(context, field) is None]
    if missing_fields:
        raise RuntimeError("Training pipeline missing fields: {}".format(", ".join(missing_fields)))
    return context


def main():
    args = apply_default_config(build_parser().parse_args())

    context = require_training_context(build_training_pipeline().handle(TrainingContext(args=args)))
    observer = build_training_observer(context)

    train(context.model, context.graph, context.ppi_list, context.loss_fn, context.loss_asl,
          context.optimizer, context.device,
          context.result_file_path, context.save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=context.scheduler,
          got=args.graph_only_train, observer=observer)


if __name__ == "__main__":
    main()
