"""Experiment choices shared by training and evaluation models."""

from typing import Literal

DatasetType = Literal['shs27k', 'shs148k', 'string']
DeviceName = Literal['auto', 'cpu', 'cuda', 'mps']
FeatureSource = Literal['global', 'local', 'both']
FusionStrategy = Literal['fixed_sum', 'concat_mlp', 'scalar', 'dynamic', 'feature_wise']
LocalEncoder = Literal['subgraph', 'sparse']
LossType = Literal['asl', 'bce']
