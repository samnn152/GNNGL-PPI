import argparse

from src.train.boolean_argument import BooleanArgument


class TrainArgumentParser:
    @staticmethod
    def build():
        parser = argparse.ArgumentParser(description='Train Model')
        parser.add_argument('--dataset_type', default='shs27k', choices=['shs27k', 'shs148k', 'string'],
                            help='dataset preset')
        parser.add_argument('--finetune_type', default='MASSA', type=str,
                            help='pretrained embedding preset name')
        parser.add_argument('--mode', default='random', type=str,
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
        parser.add_argument('--split_new', default=None, type=BooleanArgument.parse,
                            help='split new index file or not')
        parser.add_argument('--split_mode', default=None, type=str,
                            help='split method, random, bfs or dfs')
        parser.add_argument('--train_valid_index_path', default=None, type=str,
                            help='cnn_rnn and gnn unified train and valid ppi index')
        parser.add_argument('--use_lr_scheduler', default=None, type=BooleanArgument.parse,
                            help="train use learning rate scheduler or not")
        parser.add_argument('--save_path', default=None, type=str,
                            help='model save path')
        parser.add_argument('--graph_only_train', default=None, type=BooleanArgument.parse,
                            help='train ppi graph conctruct by train or all(train with test)')
        parser.add_argument('--fusion_strategy', default=None,
                            choices=['fixed_sum', 'concat_mlp', 'scalar', 'dynamic', 'feature_wise'],
                            help='global/local fusion strategy')
        parser.add_argument('--feature_source', default=None, choices=['global', 'local', 'both'],
                            help='use only the global branch, only the local branch, or both branches')
        parser.add_argument('--loss_type', default=None, choices=['asl', 'bce'],
                            help='training loss for multi-label classification')
        parser.add_argument('--subgraph_hops', default=None, type=int,
                            help='number of hops used for local subgraph extraction')
        parser.add_argument('--metrics_csv', default=None, type=str,
                            help='CSV file where train results are appended')
        parser.add_argument('--batch_size', default=None, type=int,
                            help="gnn train batch size, edge batch size")
        parser.add_argument('--epochs', default=None, type=int,
                            help='train epoch number')
        parser.add_argument('--interactive_ui', default=None, type=BooleanArgument.parse,
                            help='render an interactive terminal train dashboard')
        return parser
