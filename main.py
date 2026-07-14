# pyright: reportUnknownMemberType=false
"""Command-line entry point for training GNNGL-PPI models.

This module reads command-line arguments, resolves them into a typed training
configuration, prepares the dataset and model dependencies, runs the training
loop, and appends the final metrics to the configured CSV file.

Run training from the repository root with::

    python main.py

Inspect all available options with::

    python main.py --help

For example, train the SHS148K dataset with the global and sparse-local
branches on Apple Silicon with::

    python main.py --dataset_type shs148k --feature_source both --local_encoder sparse --device mps

Use ``--device cuda`` on a CUDA machine or ``--device auto`` to select the
best available backend automatically. Dataset paths, output paths, epoch
count, batch size, and other experiment options can be overridden through the
arguments documented by ``python main.py --help``.
"""

from src.train.controllers.train_controller import run_training


def main() -> None:
    """Prepare and execute one training run.

    The setup pipeline loads features, builds the graph, partitions the data,
    creates the model and optimizer, and prepares output paths. The resulting
    context is converted into a ``TrainingSession`` and ``TrainOptions`` before
    ``GNNTrainer.train`` starts the epoch loop. Final metrics are appended to
    the configured experiment-results CSV file.
    """
    run_training()


if __name__ == "__main__":
    main()
