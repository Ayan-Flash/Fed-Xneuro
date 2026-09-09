# Evaluation and Metrics in PS32

## 1. Overview

Federated learning systems require multidimensional evaluation spanning global performance, convergence rates, communication footprints, and cross-client fairness.

PS32 tracks the following metrics after each federated round:

---

## 2. Metric Definitions

### 2.1 Global Accuracy & Loss
- Evaluated against the centralized, hold-out test dataset (e.g. 10,000 MNIST test samples) using `calculate_accuracy` and `calculate_loss`.
- Ensures unbiased assessment of the aggregated model's generalization capabilities.

### 2.2 Convergence Trajectory
- Tracked round-by-round by `ConvergenceTracker`.
- Measures:
  - $\Delta \text{Loss}$: Step reduction in cross-entropy loss between consecutive rounds.
  - $\Delta \text{Accuracy}$: Step improvement in accuracy.
  - Milestone rounds: Number of federated rounds required to cross target performance thresholds (50%, 70%, 80%, 90%, 95%).

### 2.3 Client Fairness
- Measured by `FairnessEvaluator` across individual client test performances.
- Metrics recorded:
  - **Mean Accuracy**: $\bar{A} = \frac{1}{K} \sum_{k=1}^K A_k$
  - **Minimum Accuracy**: $\min_k A_k$
  - **Maximum Accuracy**: $\max_k A_k$
  - **Accuracy Variance**: $\sigma_A^2 = \frac{1}{K} \sum_{k=1}^K (A_k - \bar{A})^2$
  - **Accuracy Gap**: $\max_k A_k - \min_k A_k$
- Crucial for clinical applications to ensure the model does not disproportionately underperform on specific participating sites.

### 2.4 Communication Overhead
- Estimated by `CommunicationTracker` from neural network parameter sizes.
- Formulas:
  - $\text{Model Size (bytes)} = \sum_{p \in \text{params}} \text{numel}(p) \times \text{element\_size}(p)$
  - $\text{Download Bytes} = \text{Model Size} \times |S_t|$
  - $\text{Upload Bytes} = \text{Model Size} \times |S_t|$
  - $\text{Total Round Bytes} = \text{Download Bytes} + \text{Upload Bytes}$
  where $|S_t|$ is the number of clients selected in round $t$.

---

## 3. Results Persistence

At the conclusion of each simulation run, results are exported to:
- `experiments/runs/<run_id>/`:
  - `config.json`: Complete experiment parameters.
  - `metrics.json`: Detailed round-by-round metrics.
  - `metrics.csv`: Tabular metrics for data analysis.
  - `summary.json`: High-level execution summary.
- `results/metrics/<run_id>/`: Mirrors metrics for analysis pipelines.
- `results/plots/<run_id>/training_curves.png`: Rendered loss and accuracy curves.
- `models/global/<run_id>/`:
  - `round_001.pt` ... `round_010.pt`: Intermediate global checkpoints.
  - `final.pt`: Final model checkpoint.
