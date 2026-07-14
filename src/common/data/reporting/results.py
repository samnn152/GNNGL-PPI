"""CSV-backed experiment result persistence shared by train and test."""

import csv
import os
from typing import Mapping

from src.common.utils import Metrictor_PPI
from src.test.models.configuration import EvaluationConfig
from src.train.models.configuration import TrainConfig
from src.train.models.types import TrainResult


FIELDNAMES = [
    'source',
    'group',
    'variant',
    'dataset_type',
    'split_mode',
    'feature_source',
    'fusion_strategy',
    'loss_type',
    'subgraph_hops',
    'metric',
    'paper_mean_percent',
    'paper_std_percent',
    'train_loss',
    'train_recall',
    'train_precision',
    'train_f1',
    'train_time_seconds',
    'model_size_mb',
    'valid_loss',
    'valid_recall',
    'valid_precision',
    'valid_f1',
    'evaluation_split',
    'test_recall',
    'test_precision',
    'test_f1',
    'paper_reference_valid_f1_percent',
    'valid_f1_delta_percent_points',
    'best_valid_f1',
    'best_valid_epoch',
    'checkpoint_dir',
]

PAPER_GNNGL_F1_PERCENT = {
    ('shs27k', 'random'): 90.23,
    ('shs27k', 'bfs'): 78.51,
    ('shs27k', 'dfs'): 79.81,
    ('shs148k', 'random'): 93.34,
    ('shs148k', 'bfs'): 75.14,
    ('shs148k', 'dfs'): 86.07,
}


ResultValue = str | int | float | None


def ensure_csv(path: str) -> None:
    """Create the result directory and canonical CSV header when absent."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_result(path: str, row: Mapping[str, ResultValue]) -> None:
    """Append a partial result row using the canonical reporting schema."""
    ensure_csv(path)
    full_row = {field: row.get(field, '') for field in FIELDNAMES}
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(full_row)


def append_train_result(path: str, args: TrainConfig, final_stats: TrainResult) -> None:
    """Record training metrics and their delta from the matching paper result."""
    paper_f1 = PAPER_GNNGL_F1_PERCENT.get((args.dataset_type, args.split_mode))
    valid_f1_percent = final_stats.valid.f1 * 100
    delta = ''
    if paper_f1 is not None:
        delta = '{:+.4f}'.format(valid_f1_percent - paper_f1)

    append_result(path, {
        'source': 'train',
        'group': args.description,
        'variant': args.description,
        'dataset_type': args.dataset_type,
        'split_mode': args.split_mode,
        'feature_source': args.feature_source,
        'fusion_strategy': args.fusion_strategy,
        'loss_type': args.loss_type,
        'subgraph_hops': args.subgraph_hops,
        'metric': 'F1',
        'train_loss': final_stats.train.loss,
        'train_recall': final_stats.train.recall,
        'train_precision': final_stats.train.precision,
        'train_f1': final_stats.train.f1,
        'train_time_seconds': final_stats.train_time_seconds,
        'model_size_mb': final_stats.model_size_mb,
        'valid_loss': final_stats.valid.loss,
        'valid_recall': final_stats.valid.recall,
        'valid_precision': final_stats.valid.precision,
        'valid_f1': final_stats.valid.f1,
        'paper_reference_valid_f1_percent': paper_f1 if paper_f1 is not None else '',
        'valid_f1_delta_percent_points': delta,
        'best_valid_f1': final_stats.best_valid_f1,
        'best_valid_epoch': final_stats.best_valid_epoch,
        'checkpoint_dir': final_stats.save_path,
    })


def append_test_result(
    path: str,
    args: EvaluationConfig,
    evaluation_split: str,
    metrics: Metrictor_PPI,
) -> None:
    """Record evaluation metrics, checkpoint size, and paper-result delta."""
    paper_f1 = PAPER_GNNGL_F1_PERCENT.get((args.dataset_type, args.mode))
    test_f1_percent = metrics.F1 * 100
    delta = ''
    if paper_f1 is not None:
        delta = '{:+.4f}'.format(test_f1_percent - paper_f1)
    model_size_mb = ''
    if args.gnn_model and os.path.exists(args.gnn_model):
        model_size_mb = os.path.getsize(args.gnn_model) / (1024 * 1024)

    append_result(path, {
        'source': 'test',
        'group': args.description,
        'variant': args.description,
        'dataset_type': args.dataset_type,
        'split_mode': args.mode,
        'feature_source': args.feature_source,
        'fusion_strategy': args.fusion_strategy,
        'loss_type': args.loss_type,
        'subgraph_hops': args.subgraph_hops,
        'metric': 'F1',
        'evaluation_split': evaluation_split,
        'test_recall': metrics.Recall,
        'test_precision': metrics.Precision,
        'test_f1': metrics.F1,
        'model_size_mb': model_size_mb,
        'paper_reference_valid_f1_percent': paper_f1 if paper_f1 is not None else '',
        'valid_f1_delta_percent_points': delta,
        'checkpoint_dir': args.gnn_model,
    })
