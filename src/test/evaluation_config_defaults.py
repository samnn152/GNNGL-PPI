import glob
import os


class EvaluationConfigDefaults:
    DEFAULT_DATA_DIR = './assets/data'
    DEFAULT_PRETRAIN_DIR = './assets/pretrained'
    DEFAULT_INDEX_DIR = './assets/splits_{}'
    DEFAULT_OUTPUT_DIR = './outputs/save_model'
    DATASET_FILES = {
        'shs27k': (
            'protein.actions.SHS27k.STRING.txt',
            'protein.SHS27k.sequences.dictionary.tsv',
        ),
        'shs148k': (
            'protein.actions.SHS148k.STRING.txt',
            'protein.SHS148k.sequences.dictionary.tsv',
        ),
        'string': (
            '9606.protein.actions.all_connected.txt',
            'protein.STRING_all_connected.sequences.dictionary.tsv',
        ),
    }

    @classmethod
    def apply(cls, args):
        dataset_type = args.dataset_type
        mode = args.mode
        data_dir = args.data_dir or cls.DEFAULT_DATA_DIR
        pretrain_dir = args.pretrain_dir or cls.DEFAULT_PRETRAIN_DIR
        index_dir = args.index_dir or cls.DEFAULT_INDEX_DIR.format(dataset_type)
        output_dir = args.output_dir or cls.DEFAULT_OUTPUT_DIR
        ppi_file, pseq_file = cls.DATASET_FILES[dataset_type]

        args.data_dir = data_dir
        args.pretrain_dir = pretrain_dir
        args.index_dir = index_dir
        args.output_dir = output_dir

        if args.description is None:
            args.description = "test"
        if args.ppi_path is None:
            args.ppi_path = os.path.join(data_dir, ppi_file)
        if args.pseq_path is None:
            args.pseq_path = os.path.join(data_dir, pseq_file)
        if args.vec_path is None:
            args.vec_path = os.path.join(data_dir, 'vec5_CTC.txt')
        if args.pre_emb_path is None:
            args.pre_emb_path = os.path.join(pretrain_dir, 'shs_{}.pickle'.format(args.finetune_type))
        if args.index_path is None:
            args.index_path = os.path.join(index_dir, '{}.{}.fold1.json'.format(dataset_type, mode))
            fallback_index_path = './train_valid_index_json/{}.{}.fold1.json'.format(dataset_type, mode)
            if not os.path.exists(args.index_path) and os.path.exists(fallback_index_path):
                args.index_path = fallback_index_path
        if args.gnn_model is None:
            args.gnn_model = cls._default_model_path(mode, dataset_type, output_dir)
        if args.fusion_strategy is None:
            args.fusion_strategy = 'dynamic'

        return args

    @staticmethod
    def _default_model_path(mode, dataset_type, output_dir):
        default_model_path = os.path.join(
            output_dir,
            '{}_{}'.format(mode, dataset_type),
            'gnn_test_{}_{}'.format(dataset_type, mode),
            'gnn_model_valid_best.ckpt',
        )
        if os.path.exists(default_model_path):
            return default_model_path

        candidates = glob.glob(
            os.path.join(
                output_dir,
                '{}_{}'.format(mode, dataset_type),
                'gnn_test_{}_{}*'.format(dataset_type, mode),
                'gnn_model_valid_best.ckpt',
            )
        )
        return max(candidates, key=os.path.getmtime) if candidates else default_model_path
