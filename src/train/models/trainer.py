# pyright: reportUnknownMemberType=false
"""Training model for chunked PPI edge prediction."""
from __future__ import annotations

import math
import os
import random
import time

import torch
from torch import Tensor

from src.common.utils import ConsoleLogger, Metrictor_PPI
from src.train.models.types import (
    EpochStats,
    TrainObserverProtocol,
    TrainOptions,
    TrainResult,
    TrainingSession,
)


class GNNTrainer:
    """Train and validate an edge-prediction model using explicit dependencies.

    Node representations are encoded once per phase and edge predictions are
    decoded in chunks. This limits repeated GNN work while retaining a single
    accumulated gradient update per training epoch.
    """

    def __init__(self, session: TrainingSession, options: TrainOptions) -> None:
        """Bind runtime dependencies and scalar training options."""
        self.session = session
        self.options = options

    @property
    def observer(self) -> TrainObserverProtocol:
        """Expose the session observer used for progress notifications."""
        return self.session.observer

    def _log_line(self, message: str) -> None:
        if self.observer.suppress_step_logs:
            with open(self.options.result_file_path, 'a') as file:
                print(message, file=file)
            return
        ConsoleLogger.print_file(message, save_file_path=self.options.result_file_path)

    @staticmethod
    def _edge_batch(mask: list[int], step: int, batch_size: int) -> list[int]:
        start = step * batch_size
        end = start + batch_size
        return mask[start:end]

    def _batch_metrics(self, output: Tensor, label: Tensor) -> tuple[Metrictor_PPI, Tensor]:
        prediction = output.sigmoid().gt(0.5).to(dtype=torch.float32, device=self.session.device)
        metrics = Metrictor_PPI(prediction.detach().cpu(), label.detach().cpu())
        metrics.show_result()
        return metrics, prediction

    def _train_epoch(self, epoch: int) -> EpochStats:
        model = self.session.model
        graph = self.session.graph
        loss_function = self.session.loss
        optimizer = self.session.optimizer
        batch_size = self.options.batch_size

        model.train()
        self.observer.on_phase("Preparing train epoch", "Shuffling train edge masks")
        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        active_mask = graph.train_mask_got if self.options.graph_only_train else graph.train_mask
        active_edge_index = graph.edge_index_got if self.options.graph_only_train else graph.edge_index
        active_labels = graph.edge_attr_got if self.options.graph_only_train else graph.edge_attr_1
        steps = math.ceil(len(active_mask) / batch_size)
        if steps == 0:
            raise RuntimeError("Training split contains no edges")

        optimizer.zero_grad()
        self.observer.on_phase("Node encoding", "Running the GNN once for this train epoch")
        node_embeddings = model.encode_nodes(graph.x, active_edge_index, graph)
        totals = EpochStats(loss=0.0, recall=0.0, precision=0.0, f1=0.0)

        for step in range(steps):
            edge_ids = self._edge_batch(active_mask, step, batch_size)
            batch_number = step + 1
            self.observer.on_phase(
                "Train batch",
                "{}/{}: decode {} edges".format(batch_number, steps, len(edge_ids)),
            )
            output = model.decode_edges(node_embeddings, active_edge_index, edge_ids)
            label = active_labels[edge_ids].to(dtype=torch.float32, device=self.session.device)
            loss = loss_function(output, label)

            reduction = getattr(loss_function, 'reduction', 'sum')
            loss_weight = len(edge_ids) / len(active_mask) if reduction == 'mean' else 1.0 / steps
            self.observer.on_phase("Train batch", "{}/{}: accumulate gradients".format(batch_number, steps))
            (loss * loss_weight).backward(retain_graph=step < steps - 1)

            metrics, _ = self._batch_metrics(output, label)
            totals = EpochStats(
                loss=totals.loss + float(loss.item()),
                recall=totals.recall + metrics.Recall,
                precision=totals.precision + metrics.Precision,
                f1=totals.f1 + metrics.F1,
            )
            if not self.observer.suppress_step_logs:
                ConsoleLogger.print_file(
                    "epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}".format(
                        epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1
                    )
                )

        self.observer.on_phase("Optimizer", "Applying accumulated edge gradients")
        optimizer.step()
        return EpochStats(
            loss=totals.loss / steps,
            recall=totals.recall / steps,
            precision=totals.precision / steps,
            f1=totals.f1 / steps,
        )

    def _validate_epoch(self) -> EpochStats:
        model = self.session.model
        graph = self.session.graph
        batch_size = self.options.batch_size
        model.eval()

        steps = math.ceil(len(graph.val_mask) / batch_size)
        if steps == 0:
            raise RuntimeError("Validation split contains no edges")
        loss_sum = 0.0
        predictions: list[Tensor] = []
        labels: list[Tensor] = []

        with torch.no_grad():
            self.observer.on_phase("Node encoding", "Running the GNN once for validation")
            node_embeddings = model.encode_nodes(graph.x, graph.edge_index, graph)
            for step in range(steps):
                edge_ids = self._edge_batch(graph.val_mask, step, batch_size)
                output = model.decode_edges(node_embeddings, graph.edge_index, edge_ids)
                label = graph.edge_attr_1[edge_ids].to(dtype=torch.float32, device=self.session.device)
                loss = self.session.loss(output, label)
                loss_sum += float(loss.item())
                _, prediction = self._batch_metrics(output, label)
                predictions.append(prediction.cpu())
                labels.append(label.cpu())

        metrics = Metrictor_PPI(torch.cat(predictions), torch.cat(labels))
        metrics.show_result()
        return EpochStats(
            loss=loss_sum / steps,
            recall=metrics.Recall,
            precision=metrics.Precision,
            f1=metrics.F1,
        )

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        path = os.path.join(self.options.save_path, filename)
        self.observer.on_phase("Checkpoint", "Saving {}".format(path))
        torch.save({'epoch': epoch, 'state_dict': self.session.model.state_dict()}, path)

    def _step_scheduler(self, train_loss: float, epoch: int) -> None:
        scheduler = self.session.scheduler
        if scheduler is None:
            return
        scheduler.step(train_loss)
        self._log_line(
            "epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr'])
        )

    def train(self) -> TrainResult:
        """Run all epochs, checkpoint the best validation model, and summarize the run."""
        best_valid_f1 = 0.0
        best_valid_epoch = 0
        final_train_stats: EpochStats | None = None
        final_valid_stats: EpochStats | None = None
        start_time = time.time()

        self.observer.on_train_start(context=None, total_epochs=self.options.epochs)
        for epoch in range(self.options.epochs):
            self.observer.on_epoch_start(epoch)
            final_train_stats = self._train_epoch(epoch)
            if self.options.checkpoint_interval and (epoch + 1) % self.options.checkpoint_interval == 0:
                self._save_checkpoint(epoch, 'gnn_model_train.ckpt')

            final_valid_stats = self._validate_epoch()
            self._step_scheduler(final_train_stats.loss, epoch)
            if best_valid_f1 < final_valid_stats.f1:
                best_valid_f1 = final_valid_stats.f1
                best_valid_epoch = epoch
                self._save_checkpoint(epoch, 'gnn_model_valid_best.ckpt')

            self.observer.on_epoch_end(
                epoch, final_train_stats, final_valid_stats, best_valid_f1, best_valid_epoch
            )
            self._log_line(
                "epoch: {}, Train_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, "
                "Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch".format(
                    epoch,
                    final_train_stats.loss,
                    final_train_stats.recall,
                    final_train_stats.precision,
                    final_train_stats.f1,
                    final_valid_stats.loss,
                    final_valid_stats.recall,
                    final_valid_stats.precision,
                    final_valid_stats.f1,
                    best_valid_f1,
                    best_valid_epoch,
                )
            )

        self.observer.on_train_end()
        if final_train_stats is None or final_valid_stats is None:
            raise RuntimeError("Training requires at least one epoch")
        checkpoint = os.path.join(self.options.save_path, 'gnn_model_valid_best.ckpt')
        model_size_mb = os.path.getsize(checkpoint) / (1024 * 1024) if os.path.exists(checkpoint) else None
        return TrainResult(
            train=final_train_stats,
            valid=final_valid_stats,
            best_valid_f1=best_valid_f1,
            best_valid_epoch=best_valid_epoch,
            save_path=self.options.save_path,
            train_time_seconds=time.time() - start_time,
            model_size_mb=model_size_mb,
        )
