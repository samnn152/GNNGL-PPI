import argparse
import os


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def build_parser():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--dataset_type', default='shs27k', choices=['shs27k', 'shs148k', 'string'],
                        help='dataset preset')
    parser.add_argument('--finetune_type', default='MASSA', type=str,
                        help='pretrained embedding preset name')
    parser.add_argument('--mode', default='random', type=str,
                        help='split mode preset')
    parser.add_argument('--description', default=None, type=str,
                        help='train description')
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
    parser.add_argument('--split_new', default=None, type=boolean_string,
                        help='split new index file or not')
    parser.add_argument('--split_mode', default=None, type=str,
                        help='split method, random, bfs or dfs')
    parser.add_argument('--train_valid_index_path', default=None, type=str,
                        help='cnn_rnn and gnn unified train and valid ppi index')
    parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                        help="train use learning rate scheduler or not")
    parser.add_argument('--save_path', default=None, type=str,
                        help='model save path')
    parser.add_argument('--graph_only_train', default=None, type=boolean_string,
                        help='train ppi graph conctruct by train or all(train with test)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help="gnn train batch size, edge batch size")
    parser.add_argument('--epochs', default=None, type=int,
                        help='train epoch number')
    parser.add_argument('--interactive_ui', default=None, type=boolean_string,
                        help='render an interactive terminal training dashboard')
    return parser


def apply_default_config(args):
    dataset_files = {
        'shs27k': (
            './data/protein.actions.SHS27k.STRING.txt',
            './data/protein.SHS27k.sequences.dictionary.tsv',
        ),
        'shs148k': (
            './data/protein.actions.SHS148k.STRING.txt',
            './data/protein.SHS148k.sequences.dictionary.tsv',
        ),
        'string': (
            './data/9606.protein.actions.all_connected.txt',
            './data/protein.STRING_all_connected.sequences.dictionary.tsv',
        ),
    }

    mode = args.mode
    dataset_type = args.dataset_type
    ppi_path, pseq_path = dataset_files[dataset_type]

    if args.description is None:
        args.description = "test_{}_{}".format(dataset_type, mode)
    if args.ppi_path is None:
        args.ppi_path = ppi_path
    if args.pseq_path is None:
        args.pseq_path = pseq_path
    if args.vec_path is None:
        args.vec_path = './data/vec5_CTC.txt'
    if args.pre_emb_path is None:
        args.pre_emb_path = './pre_train_data/shs_{}.pickle'.format(args.finetune_type)
    if args.split_new is None:
        args.split_new = True
    if args.split_mode is None:
        args.split_mode = mode
    if args.train_valid_index_path is None:
        index_dir = './train_valid_index_json_{}/'.format(dataset_type)
        args.train_valid_index_path = os.path.join(index_dir, '{}.{}.fold1.json'.format(dataset_type, mode))
    if args.use_lr_scheduler is None:
        args.use_lr_scheduler = True
    if args.save_path is None:
        args.save_path = './save_model/{}_{}/'.format(mode, dataset_type)
    if args.graph_only_train is None:
        args.graph_only_train = False
    if args.batch_size is None:
        args.batch_size = 1024
    if args.epochs is None:
        args.epochs = 400
    if args.interactive_ui is None:
        args.interactive_ui = True

    return args
