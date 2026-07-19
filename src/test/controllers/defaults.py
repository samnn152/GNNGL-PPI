"""Resolve controller input into a complete evaluation request."""

from __future__ import annotations

import argparse
import glob
import os
from typing import ClassVar, cast

from src.test.models.configuration import EvaluationConfig
from src.common.models.configuration.experiment import DatasetType, DeviceName, FeatureSource, FusionStrategy, LocalEncoder, LossType
from src.common.models.configuration.data import SplitMode


class EvaluationConfigDefaults:
    """Resolve optional evaluation CLI values into a complete configuration."""
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
    def apply(cls, args: argparse.Namespace) -> EvaluationConfig:
        """Apply dataset-aware defaults and return an immutable evaluation request."""
        dataset_type = cast(DatasetType, args.dataset_type)
        mode = cast(SplitMode, args.mode)
        data_dir = args.data_dir or cls.DEFAULT_DATA_DIR
        pretrain_dir = args.pretrain_dir or cls.DEFAULT_PRETRAIN_DIR
        index_dir = args.index_dir or cls.DEFAULT_INDEX_DIR.format(dataset_type)
        output_dir = args.output_dir or cls.DEFAULT_OUTPUT_DIR
        ppi_file, pseq_file = cls.DATASET_FILES[dataset_type]

        index_path = args.index_path or os.path.join(index_dir, '{}.{}.fold1.json'.format(dataset_type, mode))
        fallback_index_path = './train_valid_index_json/{}.{}.fold1.json'.format(dataset_type, mode)
        if not os.path.exists(index_path) and os.path.exists(fallback_index_path):
            index_path = fallback_index_path
        finetune_type = cast(str, args.finetune_type)

        return EvaluationConfig(
            dataset_type=dataset_type,
            finetune_type=finetune_type,
            mode=mode,
            description=args.description or 'test',
            data_dir=data_dir,
            pretrain_dir=pretrain_dir,
            index_dir=index_dir,
            output_dir=output_dir,
            ppi_path=args.ppi_path or os.path.join(data_dir, ppi_file),
            pseq_path=args.pseq_path or os.path.join(data_dir, pseq_file),
            vec_path=args.vec_path or os.path.join(data_dir, 'vec5_CTC.txt'),
            pre_emb_path=args.pre_emb_path or os.path.join(pretrain_dir, 'shs_{}.pickle'.format(finetune_type)),
            index_path=index_path,
            gnn_model=args.gnn_model or cls._default_model_path(mode, dataset_type, output_dir),
            fusion_strategy=cast(FusionStrategy, args.fusion_strategy or 'feature_wise'),
            feature_source=cast(FeatureSource, args.feature_source or 'both'),
            local_encoder=cast(
                LocalEncoder,
                args.local_encoder or ('sparse' if dataset_type == 'string' else 'subgraph'),
            ),
            subgraph_hops=args.subgraph_hops if args.subgraph_hops is not None else 1,
            max_subgraph_nodes=args.max_subgraph_nodes if args.max_subgraph_nodes is not None else 64,
            max_subgraph_edges=args.max_subgraph_edges if args.max_subgraph_edges is not None else 256,
            loss_type=cast(LossType, args.loss_type or 'asl'),
            metrics_csv=args.metrics_csv or os.path.join(output_dir, 'proposal_results.csv'),
            test_batch_size=args.test_batch_size if args.test_batch_size is not None else 1024,
            device=cast(DeviceName, args.device or 'auto'),
            test_all=bool(args.test_all),
        )

    @staticmethod
    def _default_model_path(mode: SplitMode, dataset_type: DatasetType, output_dir: str) -> str:
        """Find the preferred or newest compatible best-validation checkpoint."""
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
