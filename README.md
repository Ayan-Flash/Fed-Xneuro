# PS32 — Federated Learning Simulator

PS32 is a research-grade Federated Learning (FL) simulator designed for privacy-preserving machine learning. The project lays the computational foundation for decentralized training, model aggregation, data heterogeneity benchmarking, and future multimodal explainable healthcare diagnostics (Fed-XNeuro).

---

## 1. Phase 1 Architecture

Phase 1 implements the complete local Federated Learning simulation engine using PyTorch, designed with strict modular separation:

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
          Trainer                                   Aggregation (FedAvg)
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

### Architectural Principles
1. **Server-Algorithm Decoupling**: The server manages client communication and delegates aggregation mathematics entirely to the selected algorithm (`FedAvg`).
2. **Client Isolation**: Clients train strictly on their assigned private partitions. Local training runs on independent model replicas and never mutates global state directly.
3. **Reusable Trainer**: PyTorch forward/backward/optimizer passes are encapsulated within `Trainer`, allowing future algorithms (e.g. FedProx) to inject custom loss terms without rewriting client orchestration.
4. **Independent Partitioning**: Partition strategies (IID, Label Non-IID, Dirichlet, Unbalanced) operate independently of models and algorithms.
5. **Backend Ready**: The simulation engine operates cleanly without dependencies on web frameworks or databases, ready for FastAPI and WebSocket integration in future phases.

---

## 2. Prerequisites & Environment Setup

### Prerequisites
- Python 3.11+ (Tested on Python 3.13)
- Windows PowerShell, macOS, or Linux

### Environment Setup

From the project root:

```powershell
# 1. Create the virtual environment
python -m venv backend/.venv

# 2. Activate the virtual environment
# Windows PowerShell:
.\backend\.venv\Scripts\Activate.ps1
# Linux / macOS:
source backend/.venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install -r backend/requirements.txt
```

---

## 3. How to Run the Simulator

### 3.1 Default Experiment (IID Baseline)
Runs a 5-client, 10-round FedAvg experiment on MNIST with IID data:

```powershell
python backend/run_simulation.py
```
*(Or from inside `backend/`: `python run_simulation.py`)*

### 3.2 Label Non-IID Experiment
Simulates extreme class skew (each client receives a subset of digit classes):

```powershell
python backend/run_simulation.py `
    --dataset mnist `
    --model cnn `
    --algorithm fedavg `
    --clients 5 `
    --rounds 10 `
    --local-epochs 1 `
    --partition non_iid `
    --seed 42
```

### 3.3 Dirichlet Non-IID Experiment
Simulates continuous class heterogeneity controlled by concentration parameter $\alpha$:

```powershell
python backend/run_simulation.py `
    --dataset mnist `
    --model cnn `
    --algorithm fedavg `
    --clients 5 `
    --rounds 10 `
    --local-epochs 1 `
    --partition dirichlet `
    --alpha 0.5 `
    --seed 42
```

### 3.4 Running from YAML Configuration
```powershell
python backend/run_simulation.py --config experiments/configs/baseline/fedavg_mnist_iid.yaml
```

---

## 4. Default Experiment Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dataset** | MNIST | 60k train / 10k test grayscale images (28x28) |
| **Model** | CNN | 2 Conv2D layers + 2 Linear layers (3136 -> 128 -> 10) |
| **Algorithm** | FedAvg | Sample-weighted parameter averaging |
| **Clients** | 5 | Total simulated client nodes |
| **Client Fraction** | 1.0 | 100% participation (5/5 selected each round) |
| **Rounds** | 10 | Total communication rounds |
| **Local Epochs** | 1 | Epochs trained locally per client per round |
| **Batch Size** | 32 | Local mini-batch size |
| **Learning Rate** | 0.01 | SGD learning rate |
| **Partition** | IID | Uniform and balanced data distribution |
| **Device** | Auto | CUDA if GPU available, otherwise CPU |
| **Seed** | 42 | Reproducible pseudo-random state |

---

## 5. How to Run Tests

Automated testing is managed via `pytest`:

```powershell
# From project root:
backend\.venv\Scripts\python.exe -m pytest backend/tests/ -v
```

The test suite covers:
- **FedAvg**: Mathematical verification of weighted averaging on deterministic tensors, handling integer counters, and shape validation.
- **Partitioning**: Sample coverage, absence of duplicates, balance (IID), class skew (Label Non-IID), Dirichlet $\alpha$ allocation, and unbalanced variances.
- **Client**: Model weight update verification, local evaluation metrics, parameter isolation, and state dict extraction/restoration.
- **Server**: Fractional client selection, parameter distribution, update aggregation, and full round coordination.
- **Simulation**: Rapid end-to-end integration test of the full simulation lifecycle.

---

## 6. Understanding Federated Learning Concepts

### What is FedAvg?
Federated Averaging (**FedAvg**) trains neural networks across distributed data without centralizing raw data. In each round:
1. The server broadcasts the global model $W^t$ to selected clients.
2. Each client trains locally for $E$ epochs using its private dataset.
3. The server computes the weighted average of client parameters based on client dataset sizes:
   $$W^{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_k^{t+1}$$

### IID vs. Non-IID Data
- **IID (Independent and Identically Distributed)**: Every client possesses data drawn from the same underlying distribution with approximately equal representation of all classes. In IID conditions, FedAvg converges rapidly and reliably.
- **Non-IID (Heterogeneous)**: Clients observe differing class distributions (e.g. Client 1 sees mostly digits 0 and 1; Client 2 sees digits 2 and 3). This heterogeneity induces **client drift**, causing local models to optimize toward divergent local minima and slowing global convergence.

---

## 7. Phase 2: Advanced Federated Optimization

PS32 supports advanced optimization algorithms designed to combat client drift under extreme non-IID conditions:

### 7.1 FedProx
Adds proximal regularization $\frac{\mu}{2} \|w - w_{\text{global}}\|^2$ to local loss:
```powershell
python backend/run_simulation.py --algorithm fedprox --mu 0.01 --partition dirichlet --alpha 0.1
```

### 7.2 FedAvgM
Applies server Heavy-Ball momentum to global pseudo-gradients:
```powershell
python backend/run_simulation.py --algorithm fedavgm --server-momentum 0.9 --server-lr 1.0
```

### 7.3 SCAFFOLD
Uses client and server control variates to correct local gradient drift:
```powershell
python backend/run_simulation.py --algorithm scaffold --server-lr 1.0
```

### 7.4 Multi-Algorithm Comparative Benchmark Suite
Run an automated side-by-side benchmark comparing all 4 algorithms on identical Dirichlet non-IID splits:
```powershell
python backend/benchmark.py --dataset mnist --model cnn --rounds 5 --partition dirichlet --alpha 0.1
```
Results, comparative accuracy/loss curves, and summary tables are saved to `results/plots/benchmarks/`.

---

## 8. Phase 3: Differential Privacy (DP-FL) Guard

PS32 provides complete client-level and parameter-level $(\epsilon, \delta)$-Differential Privacy:
- **$L_2$ Norm Clipping**: Bounds sensitivity by clipping client parameter deltas to threshold $C$.
- **Gaussian Perturbation**: Injects zero-mean Gaussian noise into aggregated parameters.
- **Rényi DP (RDP) Accounting**: Tracks cumulative $(\epsilon, \delta)$ spent across rounds.
- **Inversion Attack Defense**: Prevents gradient leakage and image reconstruction.

Run simulation with Differential Privacy enabled:
```powershell
python backend/run_simulation.py `
    --dataset mnist `
    --model cnn `
    --algorithm fedavg `
    --enable-dp `
    --dp-clip-norm 1.0 `
    --dp-noise-multiplier 0.5 `
    --dp-target-delta 1e-5
```

---

## 9. External Model Training & Checkpoint Resumption

You can train standalone models in another directory and connect them to PS32:
```powershell
python backend/run_simulation.py `
    --model cnn `
    --checkpoint "path/to/my_trained_model.pt" `
    --rounds 5
```

---

## 10. Documentation Index

- [FL Engine Architecture](docs/architecture/fl-architecture.md)
- [FedAvg Baseline Algorithm](docs/algorithms/fedavg.md)
- [Advanced Optimization (FedProx, FedAvgM, SCAFFOLD)](docs/algorithms/advanced_optimization.md)
- [Differential Privacy (DP-FL) Guard](docs/privacy/differential_privacy.md)
- [External Model Endpoints & Training Integration](docs/models/external_model_endpoints.md)
- [Dataset Management & Partitioning](docs/experiments/datasets.md)
- [Evaluation & Metrics Methodology](docs/experiments/evaluation.md)
