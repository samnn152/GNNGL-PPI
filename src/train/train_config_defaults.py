import os


class TrainConfigDefaults:
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
        mode = args.mode
        dataset_type = args.dataset_type
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
            args.description = "test_{}_{}".format(dataset_type, mode)
        if args.ppi_path is None:
            args.ppi_path = os.path.join(data_dir, ppi_file)
        if args.pseq_path is None:
            args.pseq_path = os.path.join(data_dir, pseq_file)
        if args.vec_path is None:
            args.vec_path = os.path.join(data_dir, 'vec5_CTC.txt')
        if args.pre_emb_path is None:
            args.pre_emb_path = os.path.join(pretrain_dir, 'shs_{}.pickle'.format(args.finetune_type))
        if args.split_new is None:
            args.split_new = True
        if args.split_mode is None:
            args.split_mode = mode
        if args.train_valid_index_path is None:
            args.train_valid_index_path = os.path.join(index_dir, '{}.{}.fold1.json'.format(dataset_type, mode))
        if args.use_lr_scheduler is None:
            args.use_lr_scheduler = True
        if args.save_path is None:
            args.save_path = os.path.join(output_dir, '{}_{}'.format(mode, dataset_type))
        if args.graph_only_train is None:
            args.graph_only_train = False
        if args.fusion_strategy is None:
            args.fusion_strategy = 'feature_wise'
        if args.feature_source is None:
            args.feature_source = 'both'
        if args.loss_type is None:
            args.loss_type = 'asl'
        if args.subgraph_hops is None:
            args.subgraph_hops = 1
        if args.metrics_csv is None:
            args.metrics_csv = os.path.join(output_dir, 'proposal_results.csv')
        if args.batch_size is None:
            args.batch_size = 1024
        if args.epochs is None:
            args.epochs = 400
        if args.interactive_ui is None:
            args.interactive_ui = True

        return args
