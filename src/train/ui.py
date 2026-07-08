import sys
import time
from dataclasses import dataclass
from shutil import get_terminal_size


@dataclass
class MetricSnapshot:
    loss: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0


class TrainObserver:
    suppress_step_logs = False

    def on_train_start(self, context, total_epochs):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_phase(self, phase, message):
        pass

    def on_epoch_end(self, epoch, train_stats, valid_stats, best_valid_f1, best_valid_epoch):
        pass

    def on_train_end(self):
        pass


class TerminalTrainUI(TrainObserver):
    suppress_step_logs = True
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"

    def __init__(self, context):
        self.context = context
        self.total_epochs = context.args.epochs
        self.current_epoch = 0
        self.completed_epochs = 0
        self.start_time = None
        self.phase = "Preparing"
        self.message = "Starting train pipeline"
        self.train_stats = MetricSnapshot()
        self.valid_stats = MetricSnapshot()
        self.previous_train_stats = None
        self.previous_valid_stats = None
        self.best_valid_f1 = 0.0
        self.best_valid_epoch = 0
        self._rendered_lines = 0
        self._last_render_time = 0.0

    def on_train_start(self, context, total_epochs):
        if context is not None:
            self.context = context
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self._render()

    def on_epoch_start(self, epoch):
        self.current_epoch = epoch
        self.phase = "Epoch"
        self.message = "Starting epoch {}".format(epoch)
        self._render()

    def on_phase(self, phase, message):
        self.phase = phase
        self.message = message
        self._render()

    def on_epoch_end(self, epoch, train_stats, valid_stats, best_valid_f1, best_valid_epoch):
        self.completed_epochs = epoch + 1
        self.previous_train_stats = self.train_stats
        self.previous_valid_stats = self.valid_stats
        self.train_stats = MetricSnapshot(train_stats.loss, train_stats.recall, train_stats.precision, train_stats.f1)
        self.valid_stats = MetricSnapshot(valid_stats.loss, valid_stats.recall, valid_stats.precision, valid_stats.f1)
        self.best_valid_f1 = best_valid_f1
        self.best_valid_epoch = best_valid_epoch
        self.phase = "Epoch complete"
        self.message = "Saved checkpoints and updated metrics"
        self._render()

    def on_train_end(self):
        self.phase = "Done"
        self.message = "Train finished"
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _elapsed(self):
        if self.start_time is None:
            return "00:00:00"
        seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    def _delta(self, current, previous, field):
        if previous is None:
            return "n/a"
        value = getattr(current, field) - getattr(previous, field)
        sign = "+" if value >= 0 else ""
        return "{}{:.4f}".format(sign, value)

    def _color(self, text, color):
        return "{}{}{}".format(color, text, self.RESET)

    def _metric_delta_color(self, metric_name, delta):
        if delta == "n/a" or delta.startswith("epoch"):
            return self.CYAN

        try:
            value = float(delta)
        except ValueError:
            return self.CYAN

        lower_is_better = metric_name.endswith("loss")
        improved = value < 0 if lower_is_better else value > 0
        if value == 0:
            return self.CYAN
        return self.GREEN if improved else self.RED

    def _phase_color(self):
        phase = self.phase.lower()
        if "validation" in phase:
            return self.CYAN
        if "checkpoint" in phase or "scheduler" in phase:
            return self.YELLOW
        if "done" in phase or "complete" in phase:
            return self.GREEN
        return self.BOLD

    def _metric_rows(self):
        rows = [
            ("train loss", self.train_stats.loss, self._delta(self.train_stats, self.previous_train_stats, "loss")),
            ("train recall", self.train_stats.recall, self._delta(self.train_stats, self.previous_train_stats, "recall")),
            ("train precision", self.train_stats.precision,
             self._delta(self.train_stats, self.previous_train_stats, "precision")),
            ("train f1", self.train_stats.f1, self._delta(self.train_stats, self.previous_train_stats, "f1")),
            ("valid loss", self.valid_stats.loss, self._delta(self.valid_stats, self.previous_valid_stats, "loss")),
            ("valid recall", self.valid_stats.recall, self._delta(self.valid_stats, self.previous_valid_stats, "recall")),
            ("valid precision", self.valid_stats.precision,
             self._delta(self.valid_stats, self.previous_valid_stats, "precision")),
            ("valid f1", self.valid_stats.f1, self._delta(self.valid_stats, self.previous_valid_stats, "f1")),
            ("best valid f1", self.best_valid_f1, "epoch {}".format(self.best_valid_epoch)),
        ]
        return rows

    def _file_rows(self):
        args = self.context.args
        graph = self.context.graph
        rows = [
            ("dataset", args.dataset_type),
            ("split mode", args.split_mode),
            ("ppi", args.ppi_path),
            ("sequence", args.pseq_path),
            ("aa vector", args.vec_path),
            ("pretrained", args.pre_emb_path),
            ("index", args.train_valid_index_path),
            ("save", self.context.save_path),
            ("fusion", args.fusion_strategy),
        ]
        if graph is not None:
            rows.extend([
                ("nodes", str(graph.num_nodes)),
                ("edges", str(graph.edge_index.shape[1])),
                ("train edges", str(len(graph.train_mask))),
                ("valid edges", str(len(graph.val_mask))),
            ])
        return rows

    def _truncate(self, text, width):
        text = str(text)
        if len(text) <= width:
            return text
        return text[:max(0, width - 3)] + "..."

    def _progress_bar(self):
        if not self.total_epochs:
            ratio = 0.0
        else:
            ratio = min(1.0, max(0.0, self.completed_epochs / self.total_epochs))
        width = 28
        filled = int(width * ratio)
        return "[{}{}] {:>5.1f}%".format("#" * filled, "-" * (width - filled), ratio * 100)

    def _fusion_alpha(self):
        model = getattr(self.context, "model", None)
        fusion = getattr(model, "global_local_fusion", None)
        if fusion is None:
            return None
        return float(fusion.alpha.detach().cpu())

    def _fusion_bar(self, width):
        alpha = self._fusion_alpha()
        if alpha is None:
            return None

        bar_width = min(32, max(10, width - 48))
        global_width = int(round(bar_width * alpha))
        local_width = bar_width - global_width
        bar = "{}{}{}".format(
            self._color("━" * global_width, self.GREEN),
            self._color("━" * local_width, self.RED),
            self.RESET,
        )
        return "fusion mean alpha  global {:>6.2%} {} local {:>6.2%}".format(
            alpha,
            bar,
            1 - alpha,
        )

    def _should_render(self, force):
        if force:
            return True
        now = time.time()
        if now - self._last_render_time >= 0.12:
            self._last_render_time = now
            return True
        return False

    def _render(self, force=False):
        if not self._should_render(force):
            return

        width = max(80, get_terminal_size((100, 30)).columns)
        content_width = width - 4
        right = "elapsed {}".format(self._elapsed())
        left = "Epoch #{} / {}".format(self.current_epoch, self.total_epochs)
        header = left + " " * max(1, content_width - len(left) - len(right)) + right

        colored_header = self._color(self._truncate(header, content_width).ljust(content_width), self.BOLD)
        colored_phase = self._color(
            self._truncate("{}: {}".format(self.phase, self.message), content_width).ljust(content_width),
            self._phase_color(),
        )

        lines = [
            "+" + "-" * (width - 2) + "+",
            "| " + colored_header + " |",
            "+" + "-" * (width - 2) + "+",
            "| " + colored_phase + " |",
            "| " + self._color(self._progress_bar().ljust(content_width), self.GREEN) + " |",
            "+" + "-" * (width - 2) + "+",
            "| " + self._color("Files and current train data".ljust(content_width), self.CYAN) + " |",
        ]

        for name, value in self._file_rows():
            row = "  {:<14} {}".format(name, value)
            lines.append("| " + self._truncate(row, content_width).ljust(content_width) + " |")

        lines.extend([
            "+" + "-" * (width - 2) + "+",
            "| " + self._color("Metrics after each epoch".ljust(content_width), self.CYAN) + " |",
            "| " + self._color(
                "{:<18} {:>14} {:>14}".format("metric", "value", "epoch delta").ljust(content_width),
                self.BOLD,
            ) + " |",
        ])

        for name, value, delta in self._metric_rows():
            row = "{:<18} {:>14.4f} {:>14}".format(name, value, delta)
            color = self._metric_delta_color(name, delta)
            lines.append("| " + self._color(self._truncate(row, content_width).ljust(content_width), color) + " |")

        fusion_row = self._fusion_bar(content_width)
        if fusion_row is not None:
            lines.append("| " + self._truncate(fusion_row, content_width).ljust(content_width) + " |")

        lines.append("+" + "-" * (width - 2) + "+")

        if self._rendered_lines:
            sys.stdout.write("\033[{}F".format(self._rendered_lines))
            sys.stdout.write("\033[J")

        sys.stdout.write("\n".join(lines) + "\n")
        self._rendered_lines = len(lines)
        sys.stdout.flush()


class TrainObserverFactory:
    @staticmethod
    def build(context):
        if not getattr(context.args, "interactive_ui", False):
            return TrainObserver()
        return TerminalTrainUI(context)
