# Federated Averaging (FedAvg)

## 1. Introduction

Federated Averaging (**FedAvg**), introduced by McMahan et al. (2017), is the foundational algorithm for decentralized deep learning. Rather than transmitting private training data to a central server, clients train copies of the global model locally on their private datasets for multiple epochs. The server then computes a sample-weighted average of the resulting client model parameters to update the global model.

---

## 2. Mathematical Formulation

Let $K$ be the total number of clients participating in round $t$.
Each client $k \in \{1, \dots, K\}$ possesses a private dataset $D_k$ with sample size $n_k = |D_k|$.
The total number of samples across all participating clients in the round is:

$$N = \sum_{k=1}^K n_k$$

### 2.1 Client Local Training
In round $t$, each selected client $k$ receives the global model parameters $W^t$ from the server.
The client initializes its local model $W_k^{t,0} = W^t$ and performs $E$ local epochs of Stochastic Gradient Descent (SGD) with mini-batch size $B$ and learning rate $\eta$:

$$W_k^{t, e+1} \leftarrow W_k^{t, e} - \eta \nabla L(W_k^{t, e}; \xi)$$

where $\xi \subset D_k$ is a mini-batch of local samples and $L$ is the loss function (Cross-Entropy Loss).
At the conclusion of local training, client $k$ obtains updated parameters $W_k^{t+1}$.

### 2.2 Server Aggregation
The server collects the parameter tensors from all participating clients and aggregates them via weighted averaging proportional to their dataset sizes:

$$W^{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_k^{t+1}$$

---

## 3. Implementation Details in PS32

In `backend/fl_engine/core/aggregator.py`, the `FedAvgAggregator` safely handles PyTorch state dictionaries:

1. **Floating Point Parameters**:
   Floating point weights and biases (`torch.float32`, `torch.float64`) are aggregated using exact weighted accumulation:
   ```python
   weighted_tensor += (param_tensor * (n_k / N)).to(ref_tensor.dtype)
   ```
2. **Non-Floating Point Tracking Tensors**:
   State dict tensors representing integer counters (such as `num_batches_tracked` in batch normalization layers) are safely handled by computing the rounded weighted average:
   ```python
   aggregated_dict[key] = torch.round(weighted_sum).to(ref_tensor.dtype)
   ```
3. **Shape and Type Integrity**:
   All tensor dimensions, parameter names, and dtypes are strictly verified against the reference global model to guarantee mathematical and structural fidelity.

---

## 4. Strengths and Limitations

### Strengths
- **Privacy Preserving**: Raw training samples never leave local client storage.
- **Communication Efficiency**: Aggregating after multiple local epochs ($E > 1$) dramatically reduces network synchronization frequency compared to distributed SGD.
- **Simplicity & Generality**: Operates natively on arbitrary PyTorch neural network architectures.

### Limitations in Non-IID Scenarios
- When local data distributions diverge significantly across clients (heterogeneous / non-IID data), local gradient trajectories drift apart ("client drift"), slowing convergence and degrading asymptotic accuracy.
- Addressed in subsequent project phases using regularization (FedProx), control variates (SCAFFOLD), and server momentum (FedAvgM).
