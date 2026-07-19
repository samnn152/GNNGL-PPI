# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
import unittest
import tempfile
from pathlib import Path
import torch

from src.train.models import GNNTrainer, TrainOptions, TrainingSession
from src.common.data.datasets.gnn_data import GNN_DATA
from src.common.data.datasets.splitting.splitter import DatasetSplitter
from src.common.data.graph.subgraph import SubgraphExtractor
from src.common.models.configuration.data import DatasetSplitConfig, GNNDataConfig
from src.common.models.configuration.graph import SparseSubgraphConfig
from src.common.models.configuration.gnn import EdgePredictionBatch, GNNModelConfig
from src.common.models.ppi_graph import PPIGraph
from src.common.models.core import GNNGL_PPI
from src.train.views import TrainObserver


class TypedConfigurationTest(unittest.TestCase):
    """Verify typed dataset and batch configuration at public model boundaries."""

    def test_dataset_constructor_uses_config_field(self) -> None:
        """Ensure the dataset parser honors fields from its typed configuration."""
        with tempfile.TemporaryDirectory() as directory:
            ppi_path = Path(directory) / 'ppi.tsv'
            ppi_path.write_text(
                'protein_a\tprotein_b\tlabel\nA\tB\tbinding\nA\tB\tactivation\n',
                encoding='utf-8',
            )
            data = GNN_DATA(GNNDataConfig(ppi_path=str(ppi_path)))

        self.assertEqual(data.node_num, 2)
        self.assertEqual(data.edge_num, 2)
        self.assertEqual(data.ppi_list, [[0, 1], [1, 0]])
        self.assertEqual(data.ppi_label_list[0], [0, 1, 0, 1, 0, 0, 0])

    def test_model_accepts_typed_batch_for_global_and_sparse_local(self) -> None:
        """Ensure supported feature branches accept the shared typed batch."""
        graph = PPIGraph(
            x=torch.randn(4, 512),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr_1=torch.zeros(4, 7),
        )
        batch = EdgePredictionBatch(graph, graph.edge_index, [0, 2])
        for source in ('global', 'both'):
            model = GNNGL_PPI(GNNModelConfig(feature_source=source, local_encoder='sparse')).eval()
            with torch.no_grad():
                output = model(batch)
            self.assertEqual(tuple(output.shape), (2, 7))
            self.assertTrue(bool(torch.isfinite(output).all()))


class SparseSubgraphExtractionTest(unittest.TestCase):
    """Verify sparse ego-graph extraction matches and bounds dense behavior."""

    def setUp(self) -> None:
        """Create a small bidirectional chain shared by extraction tests."""
        # Undirected chain 0--1--2--3, represented in both directions.
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.long,
        )

    def test_sparse_one_hop_matches_dense_extractor(self) -> None:
        """Compare unbounded one-hop sparse output with converted dense output."""
        dense_nodes, dense_edges, dense_hops = SubgraphExtractor.extract_subgraphs(
            self.edge_index,
            num_nodes=4,
            num_hops=1,
        )
        expected_nodes, expected_edges, expected_hops = SubgraphExtractor.to_sparse(
            dense_nodes,
            dense_edges,
            dense_hops,
        )
        actual_nodes, actual_edges, actual_hops = SubgraphExtractor.extract_subgraphs_sparse(
            self.edge_index,
            num_nodes=4,
            config=SparseSubgraphConfig(hops=1, max_nodes=0, max_edges=0),
        )

        self.assertTrue(torch.equal(actual_nodes[0], expected_nodes[0]))
        self.assertTrue(torch.equal(actual_nodes[1], expected_nodes[1]))
        self.assertEqual(
            set(zip(actual_edges[0].tolist(), actual_edges[1].tolist())),
            set(zip(expected_edges[0].tolist(), expected_edges[1].tolist())),
        )
        assert expected_hops is not None
        self.assertTrue(torch.equal(actual_hops, expected_hops))

    def test_sparse_extractor_bounds_each_ego_graph(self) -> None:
        """Ensure per-centroid node and edge limits are enforced."""
        nodes, edges, _ = SubgraphExtractor.extract_subgraphs_sparse(
            self.edge_index,
            num_nodes=4,
            config=SparseSubgraphConfig(hops=2, max_nodes=2, max_edges=2),
        )
        counts = torch.bincount(nodes[0], minlength=4)
        edge_counts = torch.bincount(edges[0], minlength=4)
        self.assertLessEqual(int(counts.max()), 2)
        self.assertLessEqual(int(edge_counts.max()), 2)
        for centroid in range(4):
            selected = nodes[1][nodes[0] == centroid]
            self.assertIn(centroid, selected.tolist())

    def test_sparse_combination_uses_no_dense_lookup(self) -> None:
        """Ensure lifted sparse edges use valid compact node indices."""
        nodes, edges, _ = SubgraphExtractor.extract_subgraphs_sparse(
            self.edge_index,
            num_nodes=4,
            config=SparseSubgraphConfig(hops=1, max_nodes=0, max_edges=0),
        )
        combined = SubgraphExtractor.combine_subgraphs(
            self.edge_index,
            nodes,
            edges,
            num_nodes=4,
        )
        self.assertEqual(combined.shape[0], 2)
        self.assertEqual(combined.shape[1], edges[1].numel())
        self.assertGreaterEqual(int(combined.min()), 0)
        self.assertLess(int(combined.max()), nodes[1].numel())


class DatasetSplitRatioTest(unittest.TestCase):
    """Verify independent validation and test ratios in random splitting."""

    def test_random_split_supports_separate_70_15_15_partitions(self) -> None:
        """Ensure a bidirectional graph produces the expected original-edge counts."""
        with tempfile.TemporaryDirectory() as directory:
            split = DatasetSplitter.split(
                ppi_list=[],
                edge_num=40,
                config=DatasetSplitConfig(
                    index_path=str(Path(directory) / 'split.json'),
                    validation_size=0.15,
                    test_size=0.15,
                    regenerate=True,
                    mode='random',
                ),
            )

        self.assertEqual(len(split['train_index']), 14)
        self.assertEqual(len(split['valid_index']), 3)
        self.assertEqual(len(split['test_index']), 3)


class EncodeOnceTrainerTest(unittest.TestCase):
    """Verify chunked edge loss preserves full-batch gradients and encoding reuse."""

    class DummyModel(torch.nn.Module):
        """Minimal differentiable edge model that counts node-encoding calls."""

        def __init__(self, embeddings: torch.Tensor) -> None:
            """Store trainable node embeddings and reset the encode counter."""
            super().__init__()
            self.embeddings = torch.nn.Parameter(embeddings.clone())
            self.encode_calls = 0

        def encode_nodes(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            graph: PPIGraph,
            p: float = 0.5,
        ) -> torch.Tensor:
            """Return shared node embeddings and record one encoding invocation."""
            self.encode_calls += 1
            return self.embeddings

        @staticmethod
        def decode_edges(
            node_embeddings: torch.Tensor,
            edge_index: torch.Tensor,
            edge_ids: list[int] | torch.Tensor,
        ) -> torch.Tensor:
            """Decode an edge by summing its endpoint embeddings."""
            nodes = edge_index[:, edge_ids]
            return node_embeddings[nodes[0]] + node_embeddings[nodes[1]]

    def test_bce_chunks_match_single_full_edge_objective(self) -> None:
        """Compare accumulated chunk gradients with a single full-edge BCE loss."""
        initial = torch.tensor([[0.1], [-0.2], [0.3]])
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        graph = PPIGraph(
            x=torch.empty(3, 1),
            edge_index=edge_index,
            edge_attr_1=labels,
            train_mask=[0, 1, 2, 3],
            train_mask_got=[],
        )
        model = self.DummyModel(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        trainer = GNNTrainer(
            session=TrainingSession(
                model=model,
                graph=graph,
                loss=loss_fn,
                optimizer=optimizer,
                device=torch.device('cpu'),
                observer=TrainObserver(),
            ),
            options=TrainOptions(
                result_file_path='/tmp/gnngl-test-results.txt',
                save_path='/tmp',
                batch_size=3,
                epochs=1,
            ),
        )
        trainer._train_epoch(epoch=0)

        expected_embeddings = initial.clone().requires_grad_(True)
        nodes = edge_index[:, graph.train_mask]
        expected_logits = expected_embeddings[nodes[0]] + expected_embeddings[nodes[1]]
        loss_fn(expected_logits, labels[graph.train_mask]).backward()

        self.assertEqual(model.encode_calls, 1)
        assert model.embeddings.grad is not None
        assert expected_embeddings.grad is not None
        self.assertTrue(torch.allclose(model.embeddings.grad, expected_embeddings.grad, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
