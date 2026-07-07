import argparse
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingContext:
    args: argparse.Namespace
    ppi_data: Any = None
    graph: Any = None
    ppi_list: Any = None
    device: Any = None
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    loss_fn: Any = None
    loss_asl: Any = None
    save_path: Any = None
    result_file_path: Any = None
