"""Resolve controller input into a complete training request."""

from __future__ import annotations

import argparse
import os
from typing import ClassVar, cast

from src.common.models.configuration.experiment import (
    DatasetType,
    DeviceName,
    FeatureSource,
    FusionStrategy,
    LocalEncoder,
    LossType,
)
from src.common.models.configuration.data import SplitMode
from src.train.models.configuration import TrainConfig


class TrainConfigDefaults:
    """Apply dataset presets, derived paths, and training defaults."""

    DEFAULT_DATA_DIR = './assets/data'
    DEFAULT_PRETRAIN_DIR = './assets/pretrained'
    DEFAULT_INDEX_DIR = './assets/splits_{}'
    DEFAULT_OUTPUT_DIR = './outputs/save_model'
    DATASET_FILES: ClassVar[dict[DatasetType, tuple[str, str]]] = {
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
    def apply(cls, args: argparse.Namespace) -> TrainConfig:
        """Convert a parsed CLI namespace into a validated ``TrainConfig``.

        Explicit command-line values take precedence over derived defaults.
        """
        mode = cast(SplitMode, args.mode)
        dataset_type = cast(DatasetType, args.dataset_type)
        data_dir = args.data_dir or cls.DEFAULT_DATA_DIR
        pretrain_dir = args.pretrain_dir or cls.DEFAULT_PRETRAIN_DIR
        index_dir = args.index_dir or cls.DEFAULT_INDEX_DIR.format(dataset_type)
        output_dir = args.output_dir or cls.DEFAULT_OUTPUT_DIR
        ppi_file, pseq_file = cls.DATASET_FILES[dataset_type]

        finetune_type = cast(str, args.finetune_type)
        fusion_strategy = cast(FusionStrategy, args.fusion_strategy or 'feature_wise')
        feature_source = cast(FeatureSource, args.feature_source or 'both')
        local_encoder = cast(
            LocalEncoder,
            args.local_encoder or ('sparse' if dataset_type == 'string' else 'subgraph'),
        )
        loss_type = cast(LossType, args.loss_type or 'asl')
        device = cast(DeviceName, args.device or 'auto')
        split_mode = cast(SplitMode, args.split_mode or mode)

        return TrainConfig(
            dataset_type=dataset_type,
            finetune_type=finetune_type,
            mode=mode,
            description=args.description or "test_{}_{}".format(dataset_type, mode),
            data_dir=data_dir,
            pretrain_dir=pretrain_dir,
            index_dir=index_dir,
            output_dir=output_dir,
            ppi_path=args.ppi_path or os.path.join(data_dir, ppi_file),
            pseq_path=args.pseq_path or os.path.join(data_dir, pseq_file),
            vec_path=args.vec_path or os.path.join(data_dir, 'vec5_CTC.txt'),
            pre_emb_path=args.pre_emb_path or os.path.join(pretrain_dir, 'shs_{}.pickle'.format(finetune_type)),
            go_onehot_path=args.go_onehot_path,
            pro_go_def_path=args.pro_go_def_path,
            split_new=bool(args.split_new),
            split_mode=split_mode,
            train_valid_index_path=args.train_valid_index_path
            or os.path.join(index_dir, '{}.{}.fold1.json'.format(dataset_type, mode)),
            use_lr_scheduler=bool(args.use_lr_scheduler),
            save_path=args.save_path or os.path.join(output_dir, '{}_{}'.format(mode, dataset_type)),
            graph_only_train=bool(args.graph_only_train),
            fusion_strategy=fusion_strategy,
            feature_source=feature_source,
            local_encoder=local_encoder,
            loss_type=loss_type,
            subgraph_hops=args.subgraph_hops if args.subgraph_hops is not None else 1,
            max_subgraph_nodes=args.max_subgraph_nodes if args.max_subgraph_nodes is not None else 64,
            max_subgraph_edges=args.max_subgraph_edges if args.max_subgraph_edges is not None else 256,
            metrics_csv=args.metrics_csv or os.path.join(output_dir, 'proposal_results.csv'),
            batch_size=args.batch_size if args.batch_size is not None else 1024,
            epochs=args.epochs if args.epochs is not None else 400,
            interactive_ui=bool(args.interactive_ui),
            device=device,
            checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else 0,
        )
