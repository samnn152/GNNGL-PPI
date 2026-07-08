import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.common.utils import ConsoleLogger, Metrictor_PPI
from src.train.ui import TrainObserver


@dataclass
class EpochStats:
    loss: float
    recall: float
    precision: float
    f1: float


class GNNTrainer:
    @staticmethod
    def _log_line(message, save_file_path=None, observer=None):
        if observer is not None and observer.suppress_step_logs:
            if save_file_path is not None:
                with open(save_file_path, 'a') as file:
                    print(message, file=file)
            return
        ConsoleLogger.print_file(message, save_file_path=save_file_path)

    @staticmethod
    def _edge_batch(mask, step, batch_size):
        start = step * batch_size
        end = start + batch_size
        if step == math.ceil(len(mask) / batch_size) - 1:
            return mask[start:]
        return mask[start:end]

    @staticmethod
    def _predict_from_logits(output, device):
        sigmoid = nn.Sigmoid()
        return (sigmoid(output) > 0.5).type(torch.FloatTensor).to(device)

    @classmethod
    def _batch_metrics(cls, output, label, device):
        prediction = cls._predict_from_logits(output, device)
        metrics = Metrictor_PPI(prediction.cpu().data, label.cpu().data)
        metrics.show_result()
        return metrics, prediction

    @staticmethod
    def _forward_train_batch(model, graph, edge_ids, device, got):
        if got:
            output = model(graph.x, graph.edge_index_got, edge_ids, graph.edge_go_got, graph)
            label = graph.edge_attr_got[edge_ids]
        else:
            output = model(graph.x, graph.edge_index, edge_ids, graph.edge_attr, graph)
            label = graph.edge_attr_1[edge_ids]

        return output, label.type(torch.FloatTensor).to(device)

    @staticmethod
    def _forward_valid_batch(model, graph, edge_ids, device):
        output = model(graph.x, graph.edge_index, edge_ids, graph.edge_attr, graph)
        label = graph.edge_attr_1[edge_ids]
        return output, label.type(torch.FloatTensor).to(device)

    @classmethod
    def _train_epoch(cls, model, graph, loss_asl, optimizer, device, batch_size, epoch, got, observer):
        model.train()
        observer.on_phase("Preparing train epoch", "Shuffling train edge masks")
        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        steps = math.ceil(len(graph.train_mask) / batch_size)
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        for step in range(steps):
            batch_number = step + 1
            edge_ids = cls._edge_batch(graph.train_mask_got if got else graph.train_mask, step, batch_size)
            observer.on_phase(
                "Train batch",
                "{}/{}: forward pass on {} edges".format(batch_number, steps, len(edge_ids)),
            )
            output, label = cls._forward_train_batch(model, graph, edge_ids, device, got)
            observer.on_phase(
                "Train batch",
                "{}/{}: compute ASL loss".format(batch_number, steps),
            )
            loss = loss_asl(output, label)

            observer.on_phase(
                "Train batch",
                "{}/{}: backward pass and optimizer step".format(batch_number, steps),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            observer.on_phase(
                "Train batch",
                "{}/{}: threshold logits and update running metrics".format(batch_number, steps),
            )
            metrics, _ = cls._batch_metrics(output, label, device)
            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            if not observer.suppress_step_logs:
                ConsoleLogger.print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                                         .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        return EpochStats(
            loss=loss_sum / steps,
            recall=recall_sum / steps,
            precision=precision_sum / steps,
            f1=f1_sum / steps,
        )

    @classmethod
    def _validate_epoch(cls, model, graph, loss_asl, device, batch_size, observer):
        model.eval()
        observer.on_phase("Preparing validation", "Switching model to eval mode")

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)
        valid_loss_sum = 0.0
        predictions = []
        labels = []

        with torch.no_grad():
            for step in range(valid_steps):
                batch_number = step + 1
                edge_ids = cls._edge_batch(graph.val_mask, step, batch_size)
                observer.on_phase(
                    "Validation batch",
                    "{}/{}: forward pass on {} edges".format(batch_number, valid_steps, len(edge_ids)),
                )
                output, label = cls._forward_valid_batch(model, graph, edge_ids, device)
                observer.on_phase(
                    "Validation batch",
                    "{}/{}: compute ASL validation loss".format(batch_number, valid_steps),
                )
                loss = loss_asl(output, label)
                valid_loss_sum += loss.item()

                observer.on_phase(
                    "Validation batch",
                    "{}/{}: threshold logits and collect labels".format(batch_number, valid_steps),
                )
                _, prediction = cls._batch_metrics(output, label, device)
                predictions.append(prediction.cpu().data)
                labels.append(label.cpu().data)

        valid_predictions = torch.cat(predictions, dim=0)
        valid_labels = torch.cat(labels, dim=0)
        metrics = Metrictor_PPI(valid_predictions, valid_labels)
        metrics.show_result()

        return EpochStats(
            loss=valid_loss_sum / valid_steps,
            recall=metrics.Recall,
            precision=metrics.Precision,
            f1=metrics.F1,
        )

    @staticmethod
    def _save_checkpoint(model, epoch, path, observer=None):
        if observer is not None:
            observer.on_phase("Checkpoint", "Saving {}".format(path))
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            },
            path,
        )

    @classmethod
    def _step_scheduler(cls, scheduler, train_loss, epoch, result_file_path, observer):
        if scheduler is None:
            return

        scheduler.step(train_loss)
        cls._log_line(
            "epoch: {}, now learning rate: {}".format(
                epoch,
                scheduler.optimizer.param_groups[0]['lr'],
            ),
            save_file_path=result_file_path,
            observer=observer,
        )

    @classmethod
    def train(cls, model, graph, ppi_list, loss_fn, loss_asl,
              optimizer, device,
              result_file_path, save_path,
              batch_size=512, epochs=1000, scheduler=None,
              got=False, observer=None):
        if observer is None:
            observer = TrainObserver()

        best_valid_f1 = 0.0
        best_valid_epoch = 0

        observer.on_train_start(context=None, total_epochs=epochs)
        for epoch in range(epochs):
            observer.on_epoch_start(epoch)
            train_stats = cls._train_epoch(model, graph, loss_asl, optimizer, device, batch_size, epoch, got, observer)

            cls._save_checkpoint(model, epoch, os.path.join(save_path, 'gnn_model_train.ckpt'), observer)

            valid_stats = cls._validate_epoch(model, graph, loss_asl, device, batch_size, observer)

            observer.on_phase("Scheduler", "Updating learning rate from train epoch loss")
            cls._step_scheduler(scheduler, train_stats.loss, epoch, result_file_path, observer)

            if best_valid_f1 < valid_stats.f1:
                best_valid_f1 = valid_stats.f1
                best_valid_epoch = epoch
                cls._save_checkpoint(model, epoch, os.path.join(save_path, 'gnn_model_valid_best.ckpt'), observer)

            observer.on_epoch_end(epoch, train_stats, valid_stats, best_valid_f1, best_valid_epoch)

            cls._log_line(
                "epoch: {}, Train_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, "
                "Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                .format(
                    epoch,
                    train_stats.loss,
                    train_stats.recall,
                    train_stats.precision,
                    train_stats.f1,
                    valid_stats.loss,
                    valid_stats.recall,
                    valid_stats.precision,
                    valid_stats.f1,
                    best_valid_f1,
                    best_valid_epoch,
                ),
                save_file_path=result_file_path,
                observer=observer,
            )

        observer.on_train_end()
