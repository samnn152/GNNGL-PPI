"""Seed published paper baselines into the shared result repository."""

from src.common.data.reporting.results import append_result, ensure_csv

PaperRow = tuple[str, str, str, str, str, float, float]


PAPER_ROWS: list[PaperRow] = [
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs27k', 'random', 88.21, 0.47),
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs27k', 'bfs', 63.26, 3.70),
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs27k', 'dfs', 74.31, 3.44),
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs148k', 'random', 91.98, 0.20),
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs148k', 'bfs', 65.50, 4.59),
    ('paper_table_2', 'Feature source', 'Sequence features', 'shs148k', 'dfs', 82.24, 0.73),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs27k', 'random', 88.99, 0.80),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs27k', 'bfs', 70.34, 0.67),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs27k', 'dfs', 78.32, 1.60),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs148k', 'random', 92.52, 0.17),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs148k', 'bfs', 69.99, 4.60),
    ('paper_table_2', 'Feature source', 'MASSA', 'shs148k', 'dfs', 83.19, 1.26),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs27k', 'random', 89.15, 0.69),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs27k', 'bfs', 74.08, 0.32),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs27k', 'dfs', 77.76, 2.55),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs148k', 'random', 90.84, 0.32),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs148k', 'bfs', 75.97, 3.64),
    ('paper_table_3', 'Feature source', 'Subgraph', 'shs148k', 'dfs', 82.87, 0.52),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs27k', 'random', 88.99, 0.80),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs27k', 'bfs', 70.34, 0.67),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs27k', 'dfs', 78.32, 1.60),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs148k', 'random', 92.52, 0.17),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs148k', 'bfs', 69.99, 4.60),
    ('paper_table_3', 'Feature source', 'Global Graph', 'shs148k', 'dfs', 83.19, 1.26),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs27k', 'random', 90.04, 0.55),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs27k', 'bfs', 76.08, 1.22),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs27k', 'dfs', 79.67, 4.05),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs148k', 'random', 92.76, 0.17),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs148k', 'bfs', 74.62, 4.35),
    ('paper_table_3', 'Fusion strategy', 'Combination', 'shs148k', 'dfs', 84.47, 1.20),
    ('paper_table_4', 'Subgraph size', '1-hop', 'shs27k', 'random', 90.23, 0.31),
    ('paper_table_4', 'Subgraph size', '1-hop', 'shs27k', 'bfs', 78.51, 5.25),
    ('paper_table_4', 'Subgraph size', '1-hop', 'shs27k', 'dfs', 79.81, 3.43),
    ('paper_table_4', 'Subgraph size', '2-hop', 'shs27k', 'random', 90.25, 0.25),
    ('paper_table_4', 'Subgraph size', '2-hop', 'shs27k', 'bfs', 68.29, 5.58),
    ('paper_table_4', 'Subgraph size', '2-hop', 'shs27k', 'dfs', 76.18, 1.19),
    ('paper_table_5', 'Loss function', 'BCE', 'shs27k', 'random', 90.04, 0.55),
    ('paper_table_5', 'Loss function', 'BCE', 'shs27k', 'bfs', 76.08, 1.22),
    ('paper_table_5', 'Loss function', 'BCE', 'shs27k', 'dfs', 79.67, 4.05),
    ('paper_table_5', 'Loss function', 'BCE', 'shs148k', 'random', 92.76, 0.17),
    ('paper_table_5', 'Loss function', 'BCE', 'shs148k', 'bfs', 74.62, 4.35),
    ('paper_table_5', 'Loss function', 'BCE', 'shs148k', 'dfs', 84.47, 1.20),
    ('paper_table_5', 'Loss function', 'ASL', 'shs27k', 'random', 90.23, 0.31),
    ('paper_table_5', 'Loss function', 'ASL', 'shs27k', 'bfs', 78.51, 5.25),
    ('paper_table_5', 'Loss function', 'ASL', 'shs27k', 'dfs', 79.81, 3.43),
    ('paper_table_5', 'Loss function', 'ASL', 'shs148k', 'random', 93.34, 0.12),
    ('paper_table_5', 'Loss function', 'ASL', 'shs148k', 'bfs', 75.14, 2.63),
    ('paper_table_5', 'Loss function', 'ASL', 'shs148k', 'dfs', 86.07, 1.05),
]


def seed_paper_results(path: str) -> None:
    """Append the published GNNGL-PPI ablation results to the proposal CSV."""
    ensure_csv(path)
    for source, group, variant, dataset_type, split_mode, mean, std in PAPER_ROWS:
        append_result(path, {
            'source': source,
            'group': group,
            'variant': variant,
            'dataset_type': dataset_type,
            'split_mode': split_mode,
            'metric': 'F1',
            'paper_mean_percent': mean,
            'paper_std_percent': std,
            'paper_reference_valid_f1_percent': mean,
            'valid_f1_delta_percent_points': '+0.0000',
        })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Seed proposal CSV with GNNGL-PPI paper numbers.')
    parser.add_argument('--output_csv', default='./outputs/proposal_results.csv')
    seed_paper_results(parser.parse_args().output_csv)
