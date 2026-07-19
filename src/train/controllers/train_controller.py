# pyright: reportUnknownMemberType=false
"""MVC controller for one GNNGL-PPI training run."""

import random

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
from src.train.views import TrainObserverFactory, TrainViewContext


def _build_view_context(context: TrainContext) -> TrainViewContext:
    """Map prepared model state to presentation-only terminal values."""
    args = context.args
    graph = context.require_graph()
    return TrainViewContext(
        total_epochs=args.epochs,
        file_rows=(
            ("dataset", args.dataset_type),
            ("split mode", args.split_mode),
            ("ppi", args.ppi_path),
            ("sequence", args.pseq_path),
            ("aa vector", args.vec_path),
            ("pretrained", args.pre_emb_path),
            ("index", args.train_valid_index_path),
            ("save", context.save_path),
            ("feature source", args.feature_source),
            ("local encoder", args.local_encoder),
            ("fusion", args.fusion_strategy),
            ("loss", args.loss_type),
            ("k-hop", args.subgraph_hops),
            ("nodes", str(graph.num_nodes)),
            ("edges", str(graph.edge_index.shape[1])),
            ("train edges", str(len(graph.train_mask))),
            ("valid edges", str(len(graph.val_mask))),
        ),
    )


def run_training() -> None:
    """Resolve CLI input, prepare dependencies, train, and persist metrics."""
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    config = TrainConfigDefaults.apply(TrainArgumentParser.build().parse_args())
    setup = TrainingSetupPipelineFactory.build()
    context = setup.run(TrainContext(args=config))
    observer = TrainObserverFactory.build(_build_view_context(context), config.interactive_ui)
    trainer = GNNTrainer(
        session=TrainingSession.from_context(context, observer),
        options=TrainOptions.from_context(context),
    )
    append_train_result(config.metrics_csv, config, trainer.train())
