# GNNGL-PPI

Implementation of **GNNGL-PPI: Multi-category Prediction of Protein-Protein
Interactions using Graph Neural Networks based on Global Graphs and Local
Subgraphs**.

The current codebase supports SHS27K, SHS148K, and STRING presets; global,
local, or fused node representations; random/BFS/DFS dataset splits; terminal
training monitoring; checkpoint evaluation; and CSV experiment reporting.

## Architecture

The repository uses a feature-first MVC structure with Observer events for the
live training monitor:

```text
src/
├── train/
│   ├── controllers/    CLI configuration and pipeline orchestration
│   ├── models/         setup pipeline, trainer, contracts, and result types
│   └── views/          terminal training monitor
├── test/
│   ├── controllers/    evaluation orchestration
│   ├── models/         checkpoint evaluation and analysis data
│   └── views/          analysis visualization
└── common/
    ├── data/           parsing, features, graph building, splits, and reporting
    └── models/         GNNGL-PPI, graph types, fusion, encoders, and losses
```

The primary training data flow is:

```text
CLI/View ── training options ──> Controller
Controller ── prepared graph and configuration ──> Model/Trainer
Model/Trainer ── TrainResult ──> Controller ──> experiment CSV
Model/Trainer ── observer events ──> View ──> terminal monitor
Model/Trainer ── state_dict ──> model checkpoint
```

The View does not read mutable training context or model internals directly.
The Controller prepares display-only values, while the Trainer sends progress,
metrics, and fusion balance through the Observer contract.

See the [class diagram](docs/class_diagram.md) for additional structural detail.

## Environment

The code and strict type configuration target **Python 3.10**. Runtime versions
verified with this repository are pinned in `requirements.txt`.

Core runtime dependencies are:

- PyTorch
- PyTorch Geometric
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- NumPy
- tqdm

Evaluation-analysis views additionally require:

- matplotlib
- seaborn
- LIME

Install PyTorch for the intended CPU, CUDA, or MPS backend first. Then install
PyTorch Geometric and its compiled extensions using wheels compatible with that
PyTorch build. A plain `pip install` may attempt to compile those extensions
when a matching wheel is unavailable.

Install the pinned runtime environment with:

```bash
python -m pip install -r requirements.txt
```

Do not reuse the legacy Python 3.7/DGL dependency list from the original
implementation.

For development and verification, install the runtime dependencies plus
Pyright in one command:

```bash
python -m pip install -r requirements-dev.txt
```

## Data layout

The default paths are:

```text
assets/
├── data/               PPI networks, protein sequences, and vec5_CTC.txt
├── pretrained/         MASSA/pretrained protein embeddings
├── splits_shs27k/      SHS27K split JSON files
├── splits_shs148k/     SHS148K split JSON files
└── splits_string/      STRING split JSON files

outputs/save_model/     checkpoints, logs, configuration, and result CSV
```

Dataset preparation is implemented under `src/common/data/datasets/` and
performs the following operations:

1. Parse PPI pairs and multi-label interaction types.
2. Load protein sequences.
3. Load MASSA embeddings or create deterministic fallback embeddings when the
   configured pretrained file is absent.
4. Build support sequence-vector features from `vec5_CTC.txt`.
5. Build the PyG PPI graph.
6. Load or generate train/validation/test edge indices.
7. Prepare local ego-subgraph data when the selected feature mode needs it.

### Dataset splitting

Training supports `random`, `bfs`, and `dfs` split modes.

- `--split-new` regenerates and writes the split JSON. This is enabled by
  default.
- `--no-split-new` loads the existing split JSON from
  `--train_valid_index_path`.
- Random mode supports separate validation and test ratios.
- BFS/DFS modes currently support a validation split only; `--test_size` must
  be `0`.

Example: reuse a committed split instead of regenerating it:

```bash
python main.py \
  --dataset_type shs27k \
  --mode random \
  --no-split-new
```

## Training

Inspect all supported options:

```bash
python main.py --help
```

Run training with the SHS27K defaults:

```bash
python main.py
```

Select explicit input and output roots:

```bash
python main.py \
  --dataset_type shs27k \
  --data_dir ./assets/data \
  --pretrain_dir ./assets/pretrained \
  --index_dir ./assets/splits_shs27k \
  --output_dir ./outputs/save_model \
  --device auto
```

Train the large STRING graph with the scalable sparse local encoder:

```bash
python main.py \
  --dataset_type string \
  --feature_source both \
  --local_encoder sparse \
  --batch_size 8192 \
  --device auto
```

Important model options include:

| Option | Values / behavior |
| --- | --- |
| `--feature_source` | `global`, `local`, or `both` |
| `--fusion_strategy` | `fixed_sum`, `concat_mlp`, `scalar`, `dynamic`, or `feature_wise` |
| `--local_encoder` | `subgraph` or `sparse`; STRING defaults to `sparse` |
| `--loss_type` | `asl` or `bce` |
| `--subgraph_hops` | Local ego-graph hop count |
| `--max_subgraph_nodes` | Sparse ego-graph node limit; `0` disables the limit |
| `--max_subgraph_edges` | Sparse ego-graph edge limit; `0` disables the limit |
| `--device` | `auto`, `cpu`, `cuda`, or `mps` |
| `--interactive-ui` / `--no-interactive-ui` | Enable or disable the terminal dashboard |
| `--checkpoint_interval` | Periodic train-checkpoint interval; `0` disables it |

Training and evaluation must use architecture options compatible with the
checkpoint, especially `feature_source`, `fusion_strategy`, `local_encoder`,
and local subgraph limits.

## Evaluation

Inspect evaluation options:

```bash
python main_test.py --help
```

Evaluate a specific best-validation checkpoint:

```bash
python main_test.py \
  --dataset_type shs27k \
  --mode random \
  --gnn_model "./outputs/save_model/random_shs27k/<run>/gnn_model_valid_best.ckpt" \
  --feature_source both \
  --fusion_strategy feature_wise \
  --local_encoder subgraph \
  --device auto
```

Use `--test_all True` to evaluate the complete test mask. With the default
`False`, evaluation reports non-empty test partitions grouped by how many edge
endpoints were observed during training.

Evaluation metrics are appended to the same canonical CSV schema used by
training.

## Outputs

With default arguments, each training run creates a timestamped directory:

```text
outputs/save_model/<mode>_<dataset>/gnn_<description>_<timestamp>/
├── config.txt
├── valid_results.txt
├── gnn_model_valid_best.ckpt
└── gnn_model_train.ckpt       optional; controlled by checkpoint_interval
```

Output ownership is intentionally split by responsibility:

- `GNNTrainer` writes model checkpoints and per-epoch text logs because it
  knows when validation improves and when an interval completes.
- The training/evaluation Controller receives result objects and appends the
  experiment report CSV.
- Views render terminal or analysis output; they do not own checkpoint or CSV
  persistence.

The default shared report is:

```text
outputs/save_model/proposal_results.csv
```

Override artifact locations with:

- `--output_dir`: common output root.
- `--save_path`: parent directory for timestamped training runs.
- `--metrics_csv`: training/evaluation report CSV.
- `--gnn_model`: checkpoint selected for evaluation.

## MASSA embeddings

Prepare the default MASSA embedding file with:

```bash
python -m src.common.data.pretrained.massa
```

The default destination is:

```text
assets/pretrained/shs_MASSA.pickle
```

## Verification

The following checks do not execute a full dataset training or evaluation run:

```bash
python -m compileall -q main.py main_test.py src tests
pyright
python -m unittest discover -s tests -v
python main.py --help
python main_test.py --help
```

The architecture tests protect the feature-first MVC boundaries, including the
rule that Views do not control training or read the mutable training context.

## Dataset sources

Original SHS27K and SHS148K data:

- <http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip>

Processed dataset mirror used by the original repository:

- <https://pan.baidu.com/s/1FU-Ij3LxyP9dOHZxO3Aclw> (extraction code: `tibn`)
