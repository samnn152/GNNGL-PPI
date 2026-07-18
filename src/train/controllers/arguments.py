"""Controller command-line arguments for GNNGL-PPI training.

View all options with::

    python main.py --help
"""

import argparse


class TrainArgumentParser:
    """Build the command-line interface for one training run."""

    @staticmethod
    def build() -> argparse.ArgumentParser:
        """Create the parser without reading command-line input.

        Separating construction from ``parse_args`` lets ``main`` and tests
        decide which arguments to parse. Parsed values are completed later by
        ``TrainConfigDefaults``.
        """
        parser = argparse.ArgumentParser(description='Train Model')
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
                            help='directory for train outputs and checkpoints')
        parser.add_argument('--ppi_path', default=None, type=str,
                            help="ppi path")
        parser.add_argument('--pseq_path', default=None, type=str,
                            help="protein sequence path")
        parser.add_argument('--vec_path', default=None, type=str,
                            help='protein sequence vector path')
        parser.add_argument('--pre_emb_path', default=None, type=str,
                            help='protein sequence pretrained emb path')
        parser.add_argument('--go_onehot_path', default=None, type=str,
                            help='protein sequence pretrained emb path')
        parser.add_argument('--pro_go_def_path', default=None, type=str,
                            help='protein sequence pretrained emb path')
        parser.add_argument('--split-new', '--split_new', dest='split_new', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='regenerate split index files (default: enabled)')
        parser.add_argument('--split_mode', default=None, choices=['random', 'bfs', 'dfs'],
                            help='split method, random, bfs or dfs')
        parser.add_argument('--validation_size', default=None, type=float,
                            help='fraction reserved for validation edges; default: 0.2')
        parser.add_argument('--test_size', default=None, type=float,
                            help='fraction reserved for independent test edges; default: 0')
        parser.add_argument('--train_valid_index_path', default=None, type=str,
                            help='cnn_rnn and gnn unified train and valid ppi index')
        parser.add_argument('--use-lr-scheduler', '--use_lr_scheduler', dest='use_lr_scheduler', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='use the learning-rate scheduler (default: enabled)')
        parser.add_argument('--save_path', default=None, type=str,
                            help='model save path')
        parser.add_argument('--graph-only-train', '--graph_only_train', dest='graph_only_train', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='construct the message-passing graph from training edges only')
        parser.add_argument('--fusion_strategy', default=None,
                            choices=['fixed_sum', 'concat_mlp', 'scalar', 'dynamic', 'feature_wise'],
                            help='global/local fusion strategy')
        parser.add_argument('--feature_source', default=None, choices=['global', 'local', 'both'],
                            help='use only the global branch, only the local branch, or both branches')
        parser.add_argument('--local_encoder', default=None, choices=['subgraph', 'sparse'],
                            help='local branch implementation; STRING defaults to scalable sparse GIN')
        parser.add_argument('--loss_type', default=None, choices=['asl', 'bce'],
                            help='training loss for multi-label classification')
        parser.add_argument('--subgraph_hops', default=None, type=int,
                            help='number of hops used for local subgraph extraction')
        parser.add_argument('--max_subgraph_nodes', default=None, type=int,
                            help='maximum nodes per sparse ego graph; 0 keeps the exact neighbourhood')
        parser.add_argument('--max_subgraph_edges', default=None, type=int,
                            help='maximum induced edges per sparse ego graph; 0 keeps all edges')
        parser.add_argument('--metrics_csv', default=None, type=str,
                            help='CSV file where train results are appended')
        parser.add_argument('--batch_size', default=None, type=int,
                            help="gnn train batch size, edge batch size")
        parser.add_argument('--epochs', default=None, type=int,
                            help='train epoch number')
        parser.add_argument('--interactive-ui', '--interactive_ui', dest='interactive_ui', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='render the interactive terminal dashboard (default: enabled)')
        parser.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda', 'mps'],
                            help='training device. auto prefers CUDA, then Apple Silicon MPS, then CPU')
        parser.add_argument('--checkpoint_interval', default=None, type=int,
                            help='save gnn_model_train.ckpt every N epochs. 0 disables per-epoch checkpointing')
        return parser
