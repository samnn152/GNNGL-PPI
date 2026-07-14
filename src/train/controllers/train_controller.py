# pyright: reportUnknownMemberType=false
"""MVC controller for one GNNGL-PPI training run."""

import numpy as np
import torch

from src.train.models import (
    GNNTrainer,
    TrainContext,
    TrainOptions,
    TrainingSession,
)
from src.common.data.reporting import append_train_result
from src.train.controllers.arguments import TrainArgumentParser
from src.train.controllers.defaults import TrainConfigDefaults
from src.train.controllers.setup_pipeline import TrainingSetupPipelineFactory
from src.train.views import TrainObserverFactory


def run_training() -> None:
    """Resolve CLI input, prepare dependencies, train, and persist metrics."""
    np.random.seed(1)
    torch.manual_seed(1)
    config = TrainConfigDefaults.apply(TrainArgumentParser.build().parse_args())
    setup = TrainingSetupPipelineFactory.build()
    context = setup.run(TrainContext(args=config))
    observer = TrainObserverFactory.build(context)
    trainer = GNNTrainer(
        session=TrainingSession.from_context(context, observer),
        options=TrainOptions.from_context(context),
    )
    append_train_result(config.metrics_csv, config, trainer.train())
