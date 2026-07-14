# GNNGL_PPI


Codes and models for the paper "GNNGL-PPI: Multi-category Prediction of Protein-Protein Interactions using Graph Neural Networks based on Global Graphs and Local Subgraphs".



## Using GNNGL_PPI

This repository contains:
- Requirements
- Data Processing
- Train
- Test


### Requirements
    (1) python 3.7
    (2) torch-1.10.2+cu113
    (3) torchaudio-0.10.2
    (4) torchvision-0.11.3+cu113
    (5) dgl-1.0.2+cu113
    (6) cudatoolkit-10.1.168
    (7) numpy-1.19.5
    (8) pandas
    (9) scikit-learn-0.22.2
### Data Processing

The shared dataset processing code lives in `src/common/data/datasets/gnn_data.py` (`GNN_DATA`). This package is responsible for turning raw input files into trainable graph data:

- PPI network reading (`__init__`)
- protein sequence and MASSA feature preparation (`get_feature_pretrained`)
- support sequence vectorization (`get_feature_origin`)
- PyG graph generation (`generate_data`)
- Random/BFS/DFS train-test partitioning (`split_dataset`)
    - For the first time, you need to set the parameter random_new=True to generate a new data set division json file. (Otherwise, an error will be reported, No such file or directory: "./xxxx/string.bfs.fold1.json")

`assets/data/` contains raw input files. `src/common/data/datasets/` contains Python code that parses and transforms those files.

### Train

Train entrypoint:

```bash
python main.py
```

Default runtime layout:

```text
assets/data/          PPI, sequence, and vector input files
assets/pretrained/    MASSA / pretrained protein embeddings
assets/splits_*       train/test split index files
outputs/save_model/   train checkpoints and logs
```

You can point the CLI at different input/output locations:

```bash
python main.py \
  --data_dir ./assets/data \
  --pretrain_dir ./assets/pretrained \
  --index_dir ./assets/splits_shs27k \
  --output_dir ./outputs/save_model
```

Individual files can still be overridden with `--ppi_path`, `--pseq_path`, `--vec_path`, `--pre_emb_path`, `--train_valid_index_path`, and `--save_path`.

Global/local fusion is selected with `--fusion_strategy`. For the large STRING
`all_connected` graph, the pipeline automatically uses the scalable sparse local
encoder and encodes nodes once per epoch:

```bash
python main.py \
  --dataset_type string \
  --feature_source both \
  --local_encoder sparse \
  --batch_size 8192
```

SHS datasets keep the original ego-subgraph encoder. Its memory is bounded with
`--max_subgraph_nodes` and `--max_subgraph_edges`; use `0` for either option only
when an exact, unbounded subgraph is known to fit in memory. Training and testing
must use the same `--local_encoder` and subgraph limits as the checkpoint.

Test entrypoint:

```bash
python main_test.py
```

Test uses the same directory arguments, plus `--gnn_model` when you want to evaluate a specific checkpoint.

Architecture:
- [Class diagram](docs/class_diagram.md)

### MASSA embeddings

The MASSA GNN-PPI pretrained embeddings can be prepared with:

```bash
python -m src.common.data.pretrained.massa
```

This installs `assets/pretrained/shs_MASSA.pickle`, which is used by default.


#### Dataset Download:


SHS27k and SHS148k: 
- http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip

This repositorie uses the processed dataset download path:
- https://pan.baidu.com/s/1FU-Ij3LxyP9dOHZxO3Aclw (Extraction code: tibn)
