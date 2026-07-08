import argparse

from src.test.evaluation_boolean_argument import EvaluationBooleanArgument


class EvaluationArgumentParser:
    @staticmethod
    def build():
        parser = argparse.ArgumentParser(description='Test Model')
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
        parser.add_argument('--fusion_strategy', default=None, choices=['dynamic', 'scalar'],
                            help='global/local fusion strategy used by the checkpoint')
        parser.add_argument('--test_all', default='False', type=EvaluationBooleanArgument.parse,
                            help="test all or test separately")
        return parser
