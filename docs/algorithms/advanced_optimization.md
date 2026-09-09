# Phase 2: Advanced Federated Optimization Guide

PS32 implements three state-of-the-art federated optimization algorithms designed to counter client drift and instability under non-IID partitions.

---

## 1. FedProx (Li et al., 2020)
*Federated Optimization in Heterogeneous Networks*

### Mathematical Formulation
FedProx adds a proximal regularization term to each client's local loss function, penalizing deviation from the global model received at the start of the round:

$$\min_{w} h_i(w; w^t) = F_i(w) + \frac{\mu}{2} \|w - w^t\|_2^2$$

- When $\mu = 0$, FedProx reduces to standard FedAvg.
- Larger $\mu$ restricts local model drift, crucial under severe non-IID label skew (e.g. Dirichlet $\alpha \le 0.1$).

### CLI Execution
```powershell
python backend/run_simulation.py `
    --dataset mnist `
    --model cnn `
    --algorithm fedprox `
    --mu 0.05 `
    --partition dirichlet `
    --alpha 0.1 `
    --rounds 10
```

---

## 2. FedAvgM (Hsu et al., 2019)
*Measuring the Effects of Non-Identical Distributions on Federated Visual Classification*

### Mathematical Formulation
FedAvgM applies Heavy-Ball momentum to the global pseudo-gradient on the server:

$$\Delta^{t+1} = w^t - \sum_{i \in S_t} \frac{n_i}{n} w_i^{t+1}$$
$$v^{t+1} = \beta v^t + \Delta^{t+1}$$
$$w^{t+1} = w^t - \eta_s v^{t+1}$$

Where $\beta$ is the `server_momentum` (default 0.9) and $\eta_s$ is `server_lr` (default 1.0).

### CLI Execution
```powershell
python backend/run_simulation.py `
    --algorithm fedavgm `
    --server-momentum 0.9 `
    --server-lr 1.0 `
    --rounds 10
```

---

## 3. SCAFFOLD (Karimireddy et al., 2020)
*Stochastic Controlled Averaging for Federated Learning*

### Mathematical Formulation
SCAFFOLD maintains client control variates $c_i$ and a server control variate $c$ to estimate the client drift direction and correct local gradient updates:

- **Local Update**:
  $$y_i \leftarrow y_i - \eta (\nabla F_i(y_i) - c_i + c)$$
- **Client Variate Update**:
  $$c_i^+ = c_i - c + \frac{1}{K \eta} (x - y_i)$$
  $$\Delta c_i = c_i^+ - c_i$$
- **Server Variate Aggregation**:
  $$c \leftarrow c + \frac{1}{N} \sum_{i \in S} \Delta c_i$$

### CLI Execution
```powershell
python backend/run_simulation.py `
    --algorithm scaffold `
    --server-lr 1.0 `
    --rounds 10
```

---

## 4. Automated Comparative Benchmarking Suite
To compare all algorithms under identical seeds and extreme non-IID Dirichlet splits:

```powershell
python backend/benchmark.py `
    --dataset mnist `
    --model cnn `
    --partition dirichlet `
    --alpha 0.1 `
    --rounds 5
```
This automatically produces comparison curves and a metrics summary table in `results/plots/benchmarks/`.
