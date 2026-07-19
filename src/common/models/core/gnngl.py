# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""PyTorch implementation of global-local PPI edge prediction."""
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
from torch.utils.checkpoint import checkpoint

from src.common.models.fusion import (
    ConcatMLPGlobalLocalFusionNode,
    DynamicGlobalLocalFusionNode,
    FeatureWiseGlobalLocalFusionNode,
    FixedSumGlobalLocalFusionNode,
    ScalarGlobalLocalFusionNode,
)
from src.common.models.core.gin import GINConv
from src.common.models.configuration.gnn import EdgePredictionBatch, GNNModelConfig, SubgraphKernelConfig
from src.common.models.configuration.experiment import FusionStrategy
from src.common.models.ppi_graph import PPIGraph
from src.common.models.local.subgraph import SubgraphGNNKernel

FusionFactory = Callable[[int], nn.Module]

FUSION_FACTORIES: dict[FusionStrategy, FusionFactory] = {
    'fixed_sum': lambda _hidden: FixedSumGlobalLocalFusionNode(),
    'concat_mlp': lambda hidden: ConcatMLPGlobalLocalFusionNode(hidden_size=hidden),
    'scalar': lambda _hidden: ScalarGlobalLocalFusionNode(initial_alpha=0.5),
    'dynamic': lambda hidden: DynamicGlobalLocalFusionNode(hidden_size=hidden, initial_alpha=0.5),
    'feature_wise': lambda hidden: FeatureWiseGlobalLocalFusionNode(hidden_size=hidden, initial_alpha=0.5),
}


class GNNGL_PPI(torch.nn.Module):
    """Predict PPI edge labels from global and/or local node representations.

    The global branch propagates over the complete PPI graph. The local branch
    uses either pre-extracted ego subgraphs or a sparse full-graph convolution.
    When both are enabled, the configured fusion module combines their node
    representations before edge endpoints are decoded.
    """

    def __init__(self, config: GNNModelConfig) -> None:
        """Construct model branches and the selected global-local fusion module."""
        super().__init__()
        self.config = config
        self.use_jk = config.use_jk
        self.train_eps = config.train_eps
        self.feature_fusion = config.feature_fusion
        self.feature_source = config.feature_source
        self.local_encoder = config.local_encoder
        hidden = config.hidden_size

        self.fc_x = nn.Linear(512, 512)

        self.subgraph_layers = None
        self.local_sparse_conv = None
        if config.local_encoder == 'subgraph':
            self.subgraph_layers = SubgraphGNNKernel(SubgraphKernelConfig())
        elif config.local_encoder == 'sparse':
            self.local_sparse_conv = GINConv(
                nn.Sequential(
                    nn.Linear(512, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                ),
                train_eps=self.train_eps,
            )
        else:
            raise ValueError(f"Unknown local encoder: {config.local_encoder}")
        self.norm = nn.BatchNorm1d(512)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(512, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(512, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_convs = torch.nn.ModuleList()
        for _ in range(config.num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(config.num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, config.class_count)
        self.lin1_o = nn.Linear(hidden, hidden)
        self.lin2_o = nn.Linear(hidden, hidden)
        self.global_local_fusion = self._build_global_local_fusion(config.fusion_strategy, hidden)

    @staticmethod
    def _build_global_local_fusion(fusion_strategy: FusionStrategy, hidden: int) -> nn.Module:
        """Create the configured module for combining global and local features."""
        try:
            factory = FUSION_FACTORIES[fusion_strategy]
        except KeyError as error:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}") from error
        return factory(hidden)

    def reset_parameters(self) -> None:
        """Reset trainable global-encoder and prediction-layer parameters."""
        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()
        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.fc2.reset_parameters()

    def _encode_local(self, graph: PPIGraph) -> torch.Tensor:
        """Encode node-local ego graphs through the selected local architecture."""
        if self.local_encoder == 'sparse':
            assert self.local_sparse_conv is not None
            x = self.local_sparse_conv(graph.x, graph.edge_index)
        else:
            assert self.subgraph_layers is not None
            x = self.subgraph_layers(graph)
        x = self.norm(x)
        x = F.relu(x)
        return F.dropout(x, 0.5, training=self.training)

    def _encode_global(self, x: torch.Tensor, edge_index: torch.Tensor, p: float) -> torch.Tensor:
        """Encode full-graph node features through stacked GIN convolutions."""
        x = self.fc_x(x)
        x = self.gin_conv1(x, edge_index)

        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs.append(x)

        if self.use_jk:
            x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        return self.lin2(x)

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph: PPIGraph,
        p: float = 0.5,
    ) -> torch.Tensor:
        """Encode the complete graph once before decoding edge chunks."""
        if self.feature_source == 'global':
            return self._encode_global(x, edge_index, p)
        if self.feature_source == 'local':
            return self._encode_local(graph)
        if self.feature_source == 'both':
            if self.training:
                # The branches have large, independent activation graphs. Non-
                # reentrant checkpointing keeps only their node outputs here and
                # recomputes each branch separately during backward.
                global_x = cast(torch.Tensor, checkpoint(
                    self._encode_global, x, edge_index, p, use_reentrant=False
                ))
                local_x = cast(torch.Tensor, checkpoint(
                    lambda: self._encode_local(graph), use_reentrant=False
                ))
            else:
                global_x = self._encode_global(x, edge_index, p)
                local_x = self._encode_local(graph)
            return self.global_local_fusion(global_x=global_x, local_x=local_x)
        raise ValueError("Unknown feature source: {}".format(self.feature_source))

    def decode_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Decode selected edges from reusable node representations."""
        node_id = edge_index[:, edge_ids]
        x1 = node_embeddings[node_id[0]]
        x2 = node_embeddings[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        return self.fc2(x)

    def forward(self, batch: EdgePredictionBatch, dropout: float = 0.5) -> torch.Tensor:
        """Encode batch graph nodes and predict labels for its selected edges."""
        node_embeddings = self.encode_nodes(
            batch.graph.x,
            batch.edge_index,
            batch.graph,
            p=dropout,
        )
        return self.decode_edges(node_embeddings, batch.edge_index, batch.edge_ids)
