<div align="center">

# Future-Time Temporal Path Queries

**Research code and processed data for the GRADES-NDA 2023 paper.**

<a href="https://dl.acm.org/doi/10.1145/3594778.3594879">Paper</a> ·
<a href="#method-overview">Method</a> ·
<a href="#quick-start">Quick start</a> ·
<a href="#datasets">Datasets</a>

<br>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-supported-f7931e)
![NetworkX](https://img.shields.io/badge/NetworkX-supported-376795)
![Temporal Graphs](https://img.shields.io/badge/Temporal%20Graphs-query%20processing-blueviolet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22809353.svg)](https://doi.org/10.5281/zenodo.22809353)

</div>

---

This repository implements the approach described in the paper **"Future-Time Temporal Path Queries"**.

Most temporal-graph query processing focuses on the current or previous states of a graph. This work instead studies **future-time graph queries**, where the goal is to predict the result of a query on a future graph state. The paper focuses on future-time shortest temporal path queries and combines temporal query processing with machine-learning oracles that provide information about future edges.

## Repository overview

The repository contains:

- temporal path algorithms operating on time-ordered edge streams
- link-prediction-based future path construction
- edge representations based on node embeddings
- logistic-regression link prediction with four binary embedding operators
- processed versions of the CollegeMsg, Enron, and Bitcoin Alpha datasets
- precomputed Node2vec and MultiLENS embeddings
- query and path ground truth used by the experiments

## Problem setting

A temporal graph is a directed graph $G=(V,E)$ in which every edge is represented as a triple $(u,v,t)$, where $t$ is the time at which the interaction occurs.

Let $G_{\mathrm{now}}$ contain all edges observed up to the current time, and let $\tau > \mathrm{now}$ be a future time. Given a source node $u$ and a destination node $v$, the objective is to predict the shortest temporal path between them in $G_\tau$.

A temporal path must respect time order: if its edges occur at timestamps $t_1,t_2,\ldots,t_k$, then

$$
t_1 < t_2 < \cdots < t_k.
$$

The shortest temporal path is the valid temporal path containing the fewest edges.

## Method overview

The proposed framework first processes the query on the observed graph and then invokes a prediction oracle to extend the result into the future.

### 1. Shortest temporal paths in the observed graph

The current temporal graph is processed as a stream of edges ordered by timestamp. For each node, the algorithm maintains non-dominated states containing:

- the current temporal distance from the source
- the arrival timestamp
- the corresponding temporal path

This produces the shortest temporal paths from the source to all reachable nodes in $G_{\mathrm{now}}$ using a single pass over the ordered edge stream.

### 2. Link prediction oracle

The link prediction oracle $L(w,v)$ estimates the probability that an edge from node $w$ to destination $v$ will appear during $(\mathrm{now},\tau]$.

The method considers nodes reachable from the source within the current shortest-path distance, appends a predicted edge to the destination, and returns the path with the highest predicted edge probability.

Because this oracle predicts whether an edge will appear but not its exact future timestamp, it cannot establish the temporal order of several future edges. The link-prediction method is therefore restricted to paths containing **one future edge**.

The paper evaluates link prediction using:

- **Node2vec** embeddings
- **MultiLENS** temporal embeddings
- a **logistic regression** edge classifier

### 3. Connection prediction oracle

The connection prediction oracle $C(w,t)$ predicts which node $w$ will connect to at a future timestamp $t$.

The method invokes the oracle for candidate prefix nodes and future timestamps, compares the predicted target with the query destination, and returns the best-matching temporal path together with its predicted timestamp.

The paper instantiates this oracle using **JODIE**. For comparability with the link-prediction approach, the reported experiments also restrict this setting to paths containing one future edge.

> **Repository scope:** the archive primarily contains the link-prediction experiment workflow and prepared Node2vec and MultiLENS embeddings. A standalone JODIE training implementation is not included.

## Experimental pipeline

For the link-prediction oracle, the paper uses the following temporal split:

1. Build node embeddings using the first **80%** of the interactions.
2. Use interactions in the next **10%** to train and evaluate the link classifier.
3. Recompute or load embeddings representing the first **90%** of the interactions.
4. Treat the first 90% as $G_{\mathrm{now}}$.
5. Use the final **10%** as the future evaluation interval.
6. Predict future paths and compare them with the supplied distance and path ground truth.

The edge classifier supports four ways of combining the embeddings of two endpoints:

| Operator | Edge representation |
|---|---|
| **Average** | Element-wise average |
| **Hadamard** | Element-wise product |
| **WeightedL1** | Element-wise absolute difference |
| **WeightedL2** | Element-wise squared difference |

## Repository layout

```text
.
├── Datasets/
│   ├── *_sortedEdgeStream.csv
│   ├── *_80trainSet.emb
│   ├── *_90trainSet.emb
│   ├── *_multilens_TS_s_emb.tsv
│   ├── *_testSet.csv
│   └── *_tests.csv / *_Path_tests.tsv
├── linkPrediction.py
├── predictionTest.py
├── read_Edge_Stream.py
├── predictActualDistancePath.py
├── binaryOperatorsForLearningEdgeFeatures.py
├── PerformanceMetrics.py
├── FileReader.py
├── Future-Time Temporal Path Queries.pdf
└── README.md
```

## Main files

### `linkPrediction.py`

Main experiment driver for the processed `TimestampZero` datasets included in this repository. It:

- loads the selected dataset and embeddings
- generates positive and negative edge examples
- trains logistic regression classifiers
- evaluates all four edge-embedding operators
- predicts future temporal paths
- computes path, distance, classification, and runtime statistics
- writes detailed text reports

### `predictionTest.py`

Alternative experimental driver containing configurations for additional batching and timestamp settings. Several filenames referenced by its active configuration are not included in this archive, so `linkPrediction.py` is the appropriate entry point for the bundled data.

### `read_Edge_Stream.py`

Temporal edge-stream algorithms and helper methods, including:

- earliest-arrival paths
- latest-departure paths
- fastest temporal paths
- shortest temporal path distances
- shortest temporal paths with their node sequences
- link-prediction-based distance estimation helpers

### `predictActualDistancePath.py`

Constructs and ranks future temporal paths by extending paths in the observed graph with predicted edges.

### `binaryOperatorsForLearningEdgeFeatures.py`

Creates classifier training and test data using Average, Hadamard, Weighted L1, and Weighted L2 edge representations.

### `PerformanceMetrics.py` and `FileReader.py`

Small utility modules for evaluation metrics and line-by-line file reading.

## Datasets

The paper evaluates the approach on three real temporal networks.

| Dataset | Description | Nodes | Static edges | Temporal edges | Timestamps |
|---|---|---:|---:|---:|---:|
| **CollegeMsg** | Messages exchanged in an online social network | 1,899 | 13,838 | 59,835 | 58,911 |
| **Enron** | Emails exchanged among Enron employees | 150 | 1,526 | 47,088 | 14,832 |
| **Bitcoin Alpha** | Trust-rating interactions between Bitcoin users | 3,783 | 14,124 | 24,186 | 1,647 |

The `Datasets/` directory contains processed streams, temporal train/test splits, embeddings, and query ground truth. The processed repository files may contain fewer interactions than the raw dataset statistics above because they reflect the preprocessing and node coverage used by the experiments.

## Input data

The experiment driver uses the following formats.

| File type | Required columns or structure | Purpose |
|---|---|---|
| Temporal edge stream | `source,target,time` | Chronologically ordered temporal interactions |
| Train/test edge split | `source,target` | Positive edges for classifier training or evaluation |
| Distance ground truth | `source,destination,prev_dist,future_dist` | Current and future shortest temporal distances |
| Path ground truth | `source`, `destination`, `prev_Path`, `future_Path` | Current and future temporal paths |
| Embeddings | Node identifier followed by vector values | Node representations used by the edge classifier |

The path-ground-truth files are read as tab-separated data, including the CollegeMsg file whose name ends in `.csv`.

## Quick start

### Requirements

The code is research-oriented and is not packaged. A typical environment requires:

- Python 3.9+
- `numpy`
- `pandas`
- `networkx`
- `scikit-learn`
- `torch`

PyTorch is imported by the original experiment drivers, although the bundled link classifier is implemented with scikit-learn.

Create an environment and install the dependencies:

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows PowerShell
# .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install numpy pandas networkx scikit-learn torch
```

### Run the bundled experiment

The file lists in `linkPrediction.py` use paths relative to `Datasets/`. Run the driver from that directory:

```bash
cd Datasets
python ../linkPrediction.py
```

The last uncommented dataset block in `linkPrediction.py` determines which dataset is executed. In the repository version supplied here, the active configuration is **Enron** with the precomputed Node2vec embeddings.

To run CollegeMsg or Bitcoin Alpha, edit the dataset configuration near the beginning of `linkPrediction.py` and leave exactly one dataset block active. When changing the embedding method, update both the 80% and 90% embedding-file settings consistently.

All four binary embedding operators are evaluated by default:

```python
embeddingPassingList = ["Average", "Hadamard", "WeightedL1", "WeightedL2"]
```

## Outputs

The experiment prints progress and metrics to the terminal and creates two text reports in the current working directory:

```text
<test-file-stem>__info.txt
<test-file-stem>_Multilens_Evaluation_Results.txt
```

The output includes:

- classifier training time
- AUC for the link classifier
- average future-path prediction time
- mean squared error of predicted path distances
- correctly and incorrectly predicted paths
- correctly and incorrectly predicted distances
- TP, TN, FP, and FN proportions
- minimum, maximum, and average predicted distance

The `Multilens` text in the summary filename is retained from the original implementation and is used even when the active embeddings are Node2vec embeddings.

## Results reported in the paper

The paper evaluates path correctness and distance error for the three prediction models.

| Model | Correct paths: CollegeMsg | Correct paths: Enron | Correct paths: Bitcoin | Distance MSE: CollegeMsg | Distance MSE: Enron | Distance MSE: Bitcoin |
|---|---:|---:|---:|---:|---:|---:|
| **Node2vec** | 52% | 80% | 90% | 0.64 | 0.16 | 0.20 |
| **MultiLENS** | 71% | 76% | 80% | 0.57 | 0.18 | 0.25 |
| **JODIE** | 87% | 81% | 97% | 0.42 | 0.22 | 0.04 |

JODIE also predicts the future timestamp. Its normalized timestamp MSE is reported as 0.32 for CollegeMsg, 0.15 for Enron, and 0.16 for Bitcoin Alpha.

## Reproducibility notes

- Experiment settings are defined directly in the Python files; there are no command-line arguments.
- The bundled embeddings allow evaluation without regenerating Node2vec or MultiLENS representations.
- The archive does not include the external training pipelines used to generate the embeddings or the JODIE model.
- Negative edges are sampled with Python's `random` module and no fixed seed, so exact classifier results may vary between runs.
- The implemented future-path prediction setting assumes one predicted future edge per path.
- `predictionTest.py` retains configurations for additional datasets and preprocessing variants that are not fully included in this archive.

---

## Citation

If you use this repository, please cite our [paper](https://dl.acm.org/doi/10.1145/3594778.3594879):

```bibtex
@inproceedings{gkartzios2023future,
  author    = {Gkartzios, Christos and Pitoura, Evaggelia},
  title     = {Future-Time Temporal Path Queries},
  booktitle = {Proceedings of the 6th Joint Workshop on Graph Data Management Experiences \& Systems (GRADES) and Network Data Analytics (NDA)},
  year      = {2023},
  articleno = {6},
  numpages  = {5},
  doi       = {10.1145/3594778.3594879}
}
```

---

## Contact

- Christos Gkartzios: chgartzios@cs.uoi.gr
- Evaggelia Pitoura: pitoura@uoi.gr
