# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Sparse local-subgraph extraction and mapping implementation."""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Literal, cast, overload

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul  # for propagation

from src.common.models.ppi_graph import PPIGraph
from src.common.models.configuration.graph import SparseSubgraphConfig


class SubgraphsData(PPIGraph):
    """Extend a PPI graph with PyG batching rules for lifted ego subgraphs."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Return per-field index offsets required when PyG batches graphs."""
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')] + 'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Select the concatenation dimension for lifted subgraph fields."""
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

from torch_cluster import random_walk


class SubgraphExtractor:
    """Construct dense or sparse ego-subgraph representations around each node."""

    @staticmethod
    def extract_subgraphs_sparse(
        edge_index: Tensor,
        num_nodes: int,
        config: SparseSubgraphConfig,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        """Build induced k-hop ego graphs without materialising N x N or N x E masks.

        ``max_nodes`` bounds each ego graph (including its centroid), while
        ``max_edges`` bounds its induced edges and prioritises edges incident to
        the centroid. A value of 0 disables the corresponding bound. Nodes and
        edges are selected deterministically so repeated runs are identical.
        """
        num_hops = config.hops
        max_nodes = config.max_nodes
        max_edges = config.max_edges
        if num_hops < 0:
            raise ValueError("num_hops must be non-negative")
        if max_nodes < 0:
            raise ValueError("max_nodes must be non-negative")
        if max_edges < 0:
            raise ValueError("max_edges must be non-negative")

        edge_index_cpu = edge_index.detach().cpu()
        sources = cast(list[int], edge_index_cpu[0].tolist())
        targets = cast(list[int], edge_index_cpu[1].tolist())
        neighbours: list[set[int]] = [set() for _ in range(num_nodes)]
        outgoing_edges: list[list[int]] = [[] for _ in range(num_nodes)]
        for edge_id, (source, target) in enumerate(zip(sources, targets)):
            neighbours[source].add(target)
            neighbours[target].add(source)
            outgoing_edges[source].append(edge_id)

        node_batches: list[int] = []
        node_mappers: list[int] = []
        edge_batches: list[int] = []
        edge_mappers: list[int] = []
        hop_indicators: list[int] = []

        for centroid in range(num_nodes):
            distances: dict[int, int] = {centroid: 0}
            queue = deque([centroid])
            while queue:
                node = queue.popleft()
                next_hop = distances[node] + 1
                if next_hop > num_hops:
                    continue
                for neighbour in sorted(neighbours[node]):
                    if neighbour in distances:
                        continue
                    if max_nodes and len(distances) >= max_nodes:
                        queue.clear()
                        break
                    distances[neighbour] = next_hop
                    queue.append(neighbour)

            selected_nodes = sorted(distances)
            selected_set = set(selected_nodes)
            node_batches.extend([centroid] * len(selected_nodes))
            node_mappers.extend(selected_nodes)
            hop_indicators.extend(distances[node] for node in selected_nodes)

            selected_edge_ids: list[int] = []
            if max_edges:
                # Preserve the ego-graph backbone first. With undirected input,
                # this includes both directions between centroid and neighbours.
                for source in selected_nodes:
                    for edge_id in outgoing_edges[source]:
                        target = targets[edge_id]
                        if target in selected_set and (source == centroid or target == centroid):
                            selected_edge_ids.append(edge_id)
                            if len(selected_edge_ids) >= max_edges:
                                break
                    if len(selected_edge_ids) >= max_edges:
                        break

                if len(selected_edge_ids) < max_edges:
                    selected_edge_set = set(selected_edge_ids)
                    for source in selected_nodes:
                        for edge_id in outgoing_edges[source]:
                            if edge_id in selected_edge_set:
                                continue
                            if targets[edge_id] in selected_set:
                                selected_edge_ids.append(edge_id)
                                if len(selected_edge_ids) >= max_edges:
                                    break
                        if len(selected_edge_ids) >= max_edges:
                            break
            else:
                for source in selected_nodes:
                    selected_edge_ids.extend(
                        edge_id for edge_id in outgoing_edges[source]
                        if targets[edge_id] in selected_set
                    )

            edge_batches.extend([centroid] * len(selected_edge_ids))
            edge_mappers.extend(selected_edge_ids)

        device = edge_index.device
        subgraphs_nodes = (
            torch.tensor(node_batches, dtype=torch.long, device=device),
            torch.tensor(node_mappers, dtype=torch.long, device=device),
        )
        subgraphs_edges = (
            torch.tensor(edge_batches, dtype=torch.long, device=device),
            torch.tensor(edge_mappers, dtype=torch.long, device=device),
        )
        hop_indicator = torch.tensor(hop_indicators, dtype=torch.long, device=device)
        return subgraphs_nodes, subgraphs_edges, hop_indicator

    @staticmethod
    def k_hop_subgraph(edge_index: Tensor, num_nodes: int, num_hops: int) -> tuple[Tensor, Tensor]:
        """Build dense membership and shortest-hop matrices for all node centroids."""
        print('==============k_hop_subgraph==================')
        row, col = edge_index
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(num_hops):
            next_mask = cast(Tensor, matmul(sparse_adj, hop_masks[i].float(), reduce="sum")) > 0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
        hop_indicator = hop_indicator.T
        node_mask = (hop_indicator >= 0)
        return node_mask, hop_indicator

    @staticmethod
    def combine_subgraphs(
        edge_index: Tensor,
        subgraphs_nodes: tuple[Tensor, Tensor],
        subgraphs_edges: tuple[Tensor, Tensor],
        num_nodes: int,
    ) -> Tensor:
        """Lift induced sparse edges into the combined disjoint-subgraph graph."""
        if subgraphs_edges[1].numel() == 0:
            return edge_index.new_empty((2, 0))

        # Each lifted node is uniquely identified by (subgraph, original node).
        # searchsorted maps induced edge endpoints to their lifted indices while
        # using O(number of sparse nodes) memory instead of an N x N table.
        node_keys = subgraphs_nodes[0] * num_nodes + subgraphs_nodes[1]
        order = torch.argsort(node_keys)
        sorted_keys = node_keys[order]
        selected_edges = edge_index[:, subgraphs_edges[1]]
        edge_offsets = subgraphs_edges[0] * num_nodes
        source_keys = edge_offsets + selected_edges[0]
        target_keys = edge_offsets + selected_edges[1]
        source_positions = order[torch.searchsorted(sorted_keys, source_keys)]
        target_positions = order[torch.searchsorted(sorted_keys, target_keys)]
        return torch.stack((source_positions, target_positions), dim=0)

    @staticmethod
    def random_walk_subgraph(
        edge_index: Tensor,
        num_nodes: int,
        walk_length: int,
        p: float = 1,
        q: float = 1,
        repeat: int = 1,
        cal_hops: bool = True,
        max_hops: int = 10,
    ) -> tuple[Tensor, Tensor | None]:
        """
            p (float, optional): Likelihood of immediately revisiting a node in the
                walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures
                that we are less likely to sample an already visited node in the following two steps.
            q (float, optional): Control parameter to interpolate between
                breadth-first strategy and depth-first strategy (default: :obj:`1`)
                if q > 1, the random walk is biased towards nodes close to node t.
                if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
            p, q ∈ {0.25, 0.50, 1, 2, 4}.
            Typical values:
            Fix p and tune q

            repeat: restart the random walk many times and combine together for the result

        """
        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walks = [cast(Tensor, random_walk(row, col,
                                         start=start,
                                         walk_length=walk_length,
                                         p=p, q=q,
                                         num_nodes=num_nodes)) for _ in range(repeat)]
        walk = torch.cat(walks, dim=-1)
        node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[start.repeat_interleave((walk_length + 1) * repeat), walk.reshape(-1)] = True
        if cal_hops:
            sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
            hop_indicator = row.new_full((num_nodes, num_nodes), -1)
            hop_indicator[hop_masks[0]] = 0
            for i in range(max_hops):
                next_mask = cast(Tensor, matmul(sparse_adj, hop_masks[i].float(), reduce="sum")) > 0
                hop_masks.append(next_mask)
                hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
                if hop_indicator[node_mask].min() != -1:
                    break
            return node_mask, hop_indicator
        return node_mask, None

    @staticmethod
    def to_sparse(
        node_mask: Tensor,
        edge_mask: Tensor,
        hop_indicator: Tensor | None,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor | None]:
        """Convert dense node/edge membership masks into sparse index pairs."""
        node_indices = node_mask.nonzero().T
        edge_indices = edge_mask.nonzero().T
        subgraphs_nodes = (node_indices[0], node_indices[1])
        subgraphs_edges = (edge_indices[0], edge_indices[1])
        if hop_indicator is not None:
            hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
        return subgraphs_nodes, subgraphs_edges, hop_indicator

    @staticmethod
    @overload
    def extract_subgraphs(
        edge_index: Tensor,
        num_nodes: int,
        num_hops: int,
        walk_length: int = 0,
        p: float = 1,
        q: float = 1,
        repeat: int = 1,
        *,
        sparse: Literal[False] = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Describe the dense return type selected by ``sparse=False``."""
        ...

    @staticmethod
    @overload
    def extract_subgraphs(
        edge_index: Tensor,
        num_nodes: int,
        num_hops: int,
        walk_length: int = 0,
        p: float = 1,
        q: float = 1,
        repeat: int = 1,
        *,
        sparse: Literal[True],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor | None]:
        """Describe the sparse return type selected by ``sparse=True``."""
        ...

    @staticmethod
    def extract_subgraphs(
        edge_index: Tensor,
        num_nodes: int,
        num_hops: int,
        walk_length: int = 0,
        p: float = 1,
        q: float = 1,
        repeat: int = 1,
        *,
        sparse: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None] | tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor | None]:
        """Extract all ego subgraphs using k-hop expansion or random walks."""
        print('================extract_subgraphs==============')
        if walk_length > 0:
            node_mask, hop_indicator = SubgraphExtractor.random_walk_subgraph(
                edge_index,
                num_nodes,
                walk_length,
                p=p,
                q=q,
                repeat=repeat,
                cal_hops=True,
            )
        else:
            node_mask, hop_indicator = SubgraphExtractor.k_hop_subgraph(edge_index, num_nodes, num_hops)
        edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]]
        print(node_mask.shape)
        if hop_indicator is not None:
            print(hop_indicator.shape)
        print(edge_mask.shape)
        if not sparse:
            print('sparse == False')
            return node_mask, edge_mask, hop_indicator
        return SubgraphExtractor.to_sparse(node_mask, edge_mask, hop_indicator)
