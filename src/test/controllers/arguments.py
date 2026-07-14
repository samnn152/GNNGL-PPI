"""Controller command-line arguments for evaluation."""

import argparse

from src.test.controllers.boolean import EvaluationBooleanArgument


class EvaluationArgumentParser:
    @staticmethod
    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Test Model')
        parser.add_argument('--dataset_type', default='shs27k', choices=['shs27k', 'shs148k', 'string'],
                            help='dataset preset')
        parser.add_argument('--finetune_type', default='MASSA', type=str,
                            help='pretrained embedding preset name')
        parser.add_argument('--mode', default='random', choices=['random', 'bfs', 'dfs'],
                            help='split mode preset')
        parser.add_argument('--description', default=None, type=str,
                            help='train description')
        parser.add_argument('--data_dir', default=None, type=str,
                            help='directory containing PPI, sequence, and vector input files')
        parser.add_argument('--pretrain_dir', default=None, type=str,
                            help='directory containing pretrained protein embeddings')
        parser.add_argument('--index_dir', default=None, type=str,
                            help='directory containing train/test split index files')
        parser.add_argument('--output_dir', default=None, type=str,
                            help='directory containing trained checkpoints')
        parser.add_argument('--ppi_path', default=None, type=str,
                            help="ppi path")
        parser.add_argument('--pseq_path', default=None, type=str,
                            help="protein sequence path")
        parser.add_argument('--vec_path', default=None, type=str,
                            help='protein sequence vector path')
        parser.add_argument('--pre_emb_path', default=None, type=str,
                            help='protein sequence pretrained emb path')
        parser.add_argument('--index_path', default=None, type=str,
                            help='cnn_rnn and gnn unified train and valid ppi index')
        parser.add_argument('--gnn_model', default=None, type=str,
                            help="gnn trained model")
        parser.add_argument('--fusion_strategy', default=None,
                            choices=['fixed_sum', 'concat_mlp', 'scalar', 'dynamic', 'feature_wise'],
                            help='global/local fusion strategy used by the checkpoint')
        parser.add_argument('--feature_source', default=None, choices=['global', 'local', 'both'],
                            help='feature source used by the checkpoint')
        parser.add_argument('--local_encoder', default=None, choices=['subgraph', 'sparse'],
                            help='local branch implementation used by the checkpoint')
        parser.add_argument('--subgraph_hops', default=None, type=int,
                            help='number of hops used for local subgraph extraction')
        parser.add_argument('--max_subgraph_nodes', default=None, type=int,
                            help='maximum nodes per sparse ego graph; must match training')
        parser.add_argument('--max_subgraph_edges', default=None, type=int,
                            help='maximum edges per sparse ego graph; must match training')
        parser.add_argument('--loss_type', default=None, choices=['asl', 'bce'],
                            help='loss used by the checkpoint; recorded in metrics CSV only')
        parser.add_argument('--metrics_csv', default=None, type=str,
                            help='CSV file where test results are appended')
        parser.add_argument('--test_batch_size', default=None, type=int,
                            help='edge batch size used during evaluation')
        parser.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda', 'mps'],
                            help='evaluation device. auto prefers CUDA, then Apple Silicon MPS, then CPU')
        parser.add_argument('--test_all', default='False', type=EvaluationBooleanArgument.parse,
                            help="test all or test separately")
        return parser
