# PS32 Federated Learning Engine Architecture

## 1. System Overview

PS32 is an extensible Federated Learning simulator built for privacy-preserving machine learning research. The core simulation engine executes decentralized training of deep neural networks across heterogeneous simulated clients, orchestrating model distribution, local gradient updates, client-side evaluation, and server-side aggregation.

The Phase 1 architecture strictly decouples dataset partitioning, model architectures, client execution, and federated optimization algorithms.

```text
                           PS32 FL Simulator
                                   |
                                   v
                           Simulation Engine
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
       Dataset Manager                               FL Server
             |                                           |
             v                                           v
      Data Partitioning                           Client Selection
             |                                           |
             v                                           v
        Client 1..N                               Model Distribution
             |                                           |
             v                                           v
          Trainer                                   Aggregation
             |                                           |
             +---------------------+---------------------+
                                   |
                                   v
                              Global Model
                                   |
                                   v
                               Evaluation
                                   |
                                   v
                                Metrics
                                   |
                                   v
                          Experiment Results
```

---

## 2. Component Design & Responsibilities

### 2.1 Simulation Engine (`backend/fl_engine/simulation/engine.py`)
- **Lifecycle Management**: Drives simulation states (`created`, `running`, `completed`, `failed`).
- **Orchestration**: Instantiates datasets, partitioners, models, clients, algorithm, and server.
- **Artifact Management**: Manages round-by-round checkpoints (`models/global/<run_id>/round_xxx.pt`), final models, metrics JSON/CSV, and visualization plots.

### 2.2 Federated Client (`backend/fl_engine/core/client.py`)
- Represents an edge device / medical center node with private data.
- Holds:
  - Local dataset `Subset`.
  - Independent local model replica.
  - Hyperparameters (`local_epochs`, `batch_size`, `learning_rate`, `device`).
- **Decoupling Guarantee**: The client never directly mutates or accesses the global model; it receives parameter state dictionaries, trains locally, and returns cloned CPU parameter updates.

### 2.3 Trainer (`backend/fl_engine/core/trainer.py`)
- Encapsulates local PyTorch training routines: mini-batch DataLoader generation, forward pass, loss calculation, backpropagation, optimizer stepping, and epoch iteration.
- Decoupled from `FederatedClient` to enable algorithmic training extensions (such as proximal loss terms in FedProx).

### 2.4 Federated Server (`backend/fl_engine/core/server.py`)
- Manages the authoritative global model weights.
- Handles client selection via `RandomClientSelector` using configurable fractions ($C \in (0, 1]$).
- Broadcasts current global parameters to sampled clients.
- Collects `ClientUpdate` instances (state dicts, sample counts, local metrics).
- Delegates aggregation to `BaseFederatedAlgorithm`.
- Evaluates the updated global model on the hold-out test dataset.

### 2.5 Aggregator & Algorithm Layer (`backend/fl_engine/algorithms/`)
- `BaseFederatedAlgorithm`: Abstract interface defining `aggregate(client_updates, global_parameters)`.
- `FedAvg`: Implements sample-weighted parameter averaging ($W_{global} = \sum \frac{n_k}{N} W_k$).
- Future-proof: Easily accepts new algorithms (FedProx, FedAvgM, SCAFFOLD, and the final Fed-XNeuro multimodal framework) without modifying server logic.

### 2.6 Dataset Manager & Partitioning (`backend/fl_engine/partitioning/`)
- Supports diverse data distribution scenarios:
  - **IID**: Uniform random split across clients.
  - **Label Non-IID**: Shard-based allocation inducing class imbalance.
  - **Dirichlet Non-IID**: Multinomial label allocation parameterized by concentration $\alpha$.
  - **Unbalanced**: Unequal sample size allocation.

### 2.7 Metrics & Evaluation (`backend/fl_engine/evaluation/`, `backend/fl_engine/core/metrics_manager.py`)
- Real-time tracking of top-1 classification accuracy, cross-entropy loss, convergence trajectories, and client fairness (variance and accuracy gap).
- Communication accounting: estimates byte transfer across upload and download streams based on tensor element counts.

---

## 3. Directory Layout

```text
PS32/
├── backend/
│   ├── fl_engine/
│   │   ├── algorithms/     # FL algorithms (FedAvg, FedProx, FedAvgM, SCAFFOLD)
│   │   ├── core/           # Server, Client, Trainer, Aggregator, Managers
│   │   ├── datasets/       # MNIST, CIFAR10, FashionMNIST loaders
│   │   ├── evaluation/     # Accuracy, Loss, Convergence, Fairness, Communication
│   │   ├── models/         # CNN, MLP, SmallResNet architectures
│   │   ├── partitioning/   # IID, Label Non-IID, Dirichlet, Unbalanced strategies
│   │   ├── simulation/     # Engine, Config, State, Round dataclasses
│   │   └── utils/          # Seeding, Serialization, Reproducibility diagnostics
│   ├── tests/              # Pytest suite covering all core modules
│   └── run_simulation.py   # CLI entry point
│
├── data/raw/mnist/         # Local dataset storage (git-ignored)
├── models/global/          # Saved model checkpoints per experiment run (git-ignored)
├── experiments/            # YAML baseline configs and run manifests
└── results/                # Output metrics (JSON, CSV) and training plots
```
