# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
"""Local ego-subgraph encoder implementation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from src.common.models.ppi_graph import PPIGraph
from src.common.models.configuration.gnn import SubgraphKernelConfig
from src.common.models.local.elements import Identity, MLP


class LocalGNN(nn.Module):
    """Apply one message-passing layer to the lifted nodes of local subgraphs."""

    def __init__(self, config: SubgraphKernelConfig) -> None:
        """Build the local convolution, normalization, and output projection."""
        super().__init__()
        hop_size = config.hop_embedding_size
        hidden_input_size = config.input_size + hop_size
        self.conv = GINConv(
            nn.Sequential(nn.Linear(hidden_input_size, hidden_input_size)),
            train_eps=True,
        )
        self.norm: nn.Module = nn.BatchNorm1d(hidden_input_size)
        self.output_encoder: nn.Module = (
            MLP(hidden_input_size, config.output_size, layer_count=1, with_final_activation=False, bias=config.bias)
            if hidden_input_size != config.output_size else Identity()
        )
        self.dropout = config.dropout
        self.residual = config.residual

    def reset_parameters(self) -> None:
        """Reset all trainable local message-passing parameters."""
        self.output_encoder.reset_parameters()  # type: ignore[reportAttributeAccessIssue]
        self.conv.reset_parameters()
        self.norm.reset_parameters()  # type: ignore[reportAttributeAccessIssue]

    def forward(self, inputs: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Encode lifted subgraph nodes while preserving the PyG call signature."""
        del batch  # Kept for the PyG kernel call shape.
        output = F.relu(self.norm(self.conv(inputs, edge_index)))
        output = F.dropout(output, self.dropout, training=self.training)
        if self.residual:
            output = output + inputs
        return self.output_encoder(output)


class SubgraphGNNKernel(nn.Module):
    """Aggregate bounded ego-subgraphs into one local vector per graph node.

    Lifted nodes receive hop embeddings, pass through a local GNN, and are
    pooled as configured centroid, subgraph, and/or context representations.
    """

    def __init__(self, config: SubgraphKernelConfig) -> None:
        """Construct hop embeddings, local transforms, gates, and aggregators."""
        super().__init__()
        if not config.embeddings or min(config.embeddings) < 0 or max(config.embeddings) > 2:
            raise ValueError("embeddings must contain values from 0 through 2")
        self.config = config
        self.hop_embedder = nn.Embedding(20, config.hop_embedding_size)
        self.gnn = LocalGNN(config)
        self.subgraph_transform = MLP(
            config.output_size, config.output_size,
            layer_count=config.mlp_layers, with_final_activation=True,
        )
        self.context_transform = MLP(
            config.output_size, config.output_size,
            layer_count=config.mlp_layers, with_final_activation=True,
        )
        combined_size = (
            config.output_size
            if config.combine_mode == 'add'
            else config.output_size * len(config.embeddings)
        )
        self.out_encoder = MLP(
            combined_size, config.output_size,
            layer_count=config.mlp_layers, with_final_activation=False,
            bias=config.bias, with_norm=True,
        )
        self.gate_mapper_subgraph = nn.Sequential(
            nn.Linear(config.hop_embedding_size, config.output_size), nn.Sigmoid()
        )
        self.gate_mapper_context = nn.Sequential(
            nn.Linear(config.hop_embedding_size, config.output_size), nn.Sigmoid()
        )
        self.gate_mapper_centroid = nn.Sequential(
            nn.Linear(config.hop_embedding_size, config.output_size), nn.Sigmoid()
        )

    def reset_parameters(self) -> None:
        """Reset every embedding, GNN, projection, and gating module."""
        self.hop_embedder.reset_parameters()
        self.gnn.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data: PPIGraph) -> Tensor:
        """Encode precomputed subgraph mappings into node-aligned local vectors."""
        config = self.config
        lifted_x = data.x[data.subgraphs_nodes_mapper]
        hop_embedding = self.hop_embedder(data.hop_indicator + 1)
        lifted_x = torch.cat([lifted_x, hop_embedding], dim=-1)
        lifted_x = self.gnn(lifted_x, data.combined_subgraphs, data.subgraphs_batch)
        centroid_mask = data.subgraphs_nodes_mapper == data.subgraphs_batch
        embeddings: dict[int, Tensor] = {}

        if 0 in config.embeddings:
            centroid_x = lifted_x[centroid_mask]
            embeddings[0] = centroid_x * self.gate_mapper_centroid(hop_embedding[centroid_mask])
        if 1 in config.embeddings:
            subgraph_x = (
                self.subgraph_transform(F.dropout(lifted_x, config.dropout, training=self.training))
                if len(config.embeddings) > 1 else lifted_x
            )
            subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_embedding)
            embeddings[1] = scatter(
                subgraph_x, data.subgraphs_batch, dim=0, reduce=config.pooling
            )
        if 2 in config.embeddings:
            context_x = (
                self.context_transform(F.dropout(lifted_x, config.dropout, training=self.training))
                if len(config.embeddings) > 1 else lifted_x
            )
            context_x = context_x * self.gate_mapper_context(hop_embedding)
            embeddings[2] = scatter(
                context_x, data.subgraphs_nodes_mapper, dim=0, reduce=config.pooling
            )

        selected = [embeddings[index] for index in config.embeddings]
        if config.combine_mode == 'add':
            return torch.stack(selected, dim=0).sum(dim=0)
        return self.out_encoder(torch.cat(selected, dim=-1))
