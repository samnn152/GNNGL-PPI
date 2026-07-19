from __future__ import annotations

import random

from torch import Tensor


class ConsoleLogger:
    """Write the same diagnostic message to stdout and an optional file."""

    @staticmethod
    def print_file(message: object, save_file_path: str | None = None) -> None:
        """Print a message and append it to ``save_file_path`` when supplied."""
        print(message)
        if save_file_path is not None:
            with open(save_file_path, 'a') as file:
                print(message, file=file)


class Metrictor_PPI:
    """Calculate flattened binary classification metrics for PPI labels."""

    def __init__(self, prediction: Tensor, truth: Tensor, is_binary: bool = False) -> None:
        """Calculate confusion counts and derived scores from predictions."""
        del is_binary  # Both legacy branches flatten identically.
        prediction = prediction.reshape(-1)
        truth = truth.reshape(-1)
        matches = prediction == truth
        positives = truth == 1
        predicted_positives = prediction == 1
        self.TP = int((matches & positives).sum().item())
        self.TN = int((matches & ~positives).sum().item())
        self.FN = int((~matches & positives).sum().item())
        self.FP = int((~matches & predicted_positives).sum().item())
        self.num = int(truth.numel())
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)

    def show_result(self, is_print: bool = False, file: str | None = None) -> None:
        """Print calculated scores when explicitly enabled by the caller."""
        if not is_print:
            return
        ConsoleLogger.print_file(f"Accuracy: {self.Accuracy}", file)
        ConsoleLogger.print_file(f"Precision: {self.Precision}", file)
        ConsoleLogger.print_file(f"Recall: {self.Recall}", file)
        ConsoleLogger.print_file(f"F1-Score: {self.F1}", file)


class UnionFindSet:
    """Track disjoint graph components using union by rank and path compression."""

    def __init__(self, member_count: int) -> None:
        """Create one independent component for each member."""
        self.roots = list(range(member_count))
        self.rank = [0] * member_count
        self.count = member_count

    def find(self, member: int) -> int:
        """Return a member's component root while compressing its traversed path."""
        path: list[int] = []
        while member != self.roots[member]:
            path.append(member)
            member = self.roots[member]
        for root in path:
            self.roots[root] = member
        return member

    def union(self, first: int, second: int) -> None:
        """Merge two member components when they are currently disjoint."""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] > self.rank[second_root]:
            self.roots[second_root] = first_root
        elif self.rank[first_root] < self.rank[second_root]:
            self.roots[first_root] = second_root
        else:
            self.roots[second_root] = first_root
            self.rank[first_root] -= 1
        self.count -= 1


class GraphSplitSampler:
    """Select connected validation-edge subsets through graph traversal."""

    @staticmethod
    def get_bfs_sub_graph(
        ppi_list: list[list[int]],
        node_num: int,
        node_to_edge_index: dict[int, list[int]],
        sub_graph_size: int,
    ) -> list[int]:
        """Collect up to ``sub_graph_size`` edges in breadth-first order."""
        candidate_nodes: list[int] = []
        selected_edges: list[int] = []
        selected_nodes: list[int] = []
        random_node = random.randint(0, node_num - 1)
        while len(node_to_edge_index[random_node]) > 5:
            random_node = random.randint(0, node_num - 1)
        candidate_nodes.append(random_node)
        while len(selected_edges) < sub_graph_size and candidate_nodes:
            current_node = candidate_nodes.pop(0)
            selected_nodes.append(current_node)
            for edge_index in node_to_edge_index[current_node]:
                if edge_index in selected_edges:
                    continue
                selected_edges.append(edge_index)
                source, target = ppi_list[edge_index]
                end_node = target if source == current_node else source
                if end_node not in selected_nodes and end_node not in candidate_nodes:
                    candidate_nodes.append(end_node)
        return selected_edges

    @staticmethod
    def get_dfs_sub_graph(
        ppi_list: list[list[int]],
        node_num: int,
        node_to_edge_index: dict[int, list[int]],
        sub_graph_size: int,
    ) -> list[int]:
        """Collect up to ``sub_graph_size`` edges in depth-first order."""
        stack: list[int] = []
        selected_edges: list[int] = []
        selected_nodes: list[int] = []
        random_node = random.randint(0, node_num - 1)
        while len(node_to_edge_index[random_node]) > 5:
            random_node = random.randint(0, node_num - 1)
        stack.append(random_node)
        while len(selected_edges) < sub_graph_size and stack:
            current_node = stack[-1]
            if current_node in selected_nodes:
                next_nodes: list[int] = []
                for edge_index in node_to_edge_index[current_node]:
                    source, target = ppi_list[edge_index]
                    end_node = target if source == current_node else source
                    if end_node not in selected_nodes:
                        next_nodes.append(end_node)
                        break
                if next_nodes:
                    stack.extend(next_nodes)
                else:
                    stack.pop()
                continue
            selected_nodes.append(current_node)
            selected_edges.extend(
                edge_index
                for edge_index in node_to_edge_index[current_node]
                if edge_index not in selected_edges
            )
        return selected_edges[:sub_graph_size]
