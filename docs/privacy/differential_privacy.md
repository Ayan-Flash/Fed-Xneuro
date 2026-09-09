# Phase 3: Differential Privacy (DP-FL) Guard

PS32 features a comprehensive client-level and parameter-level Differential Privacy (DP) system designed to safeguard patient records, biomarkers, and neuroimaging scans against privacy leakage.

---

## 1. Threat Model & Privacy Guarantees
In clinical federated learning for MCI and Alzheimer's progression:
- An honest-but-curious server or network adversary can reconstruct training images and patient features from model updates (Gradient Inversion Attack / Deep Leakage).
- **PS32 Solution**: Implements $(\epsilon, \delta)$-Differential Privacy via norm clipping and calibrated Gaussian perturbation.

---

## 2. Core DP Mechanisms

### Step 1: Client-Level $L_2$ Update Clipping
To bound the sensitivity of each client's contribution, updates $\Delta_i = w_i - w_{\text{global}}$ are clipped to maximum $L_2$ threshold $C$:

$$\Delta_i \leftarrow \Delta_i \cdot \min\left(1, \frac{C}{\|\Delta_i\|_2}\right)$$

### Step 2: Calibrated Gaussian Perturbation
Zero-mean Gaussian noise is added to the aggregated global model:

$$w^{t+1} \leftarrow \text{Agg}(\{w_i\}) + \mathcal{N}\left(0, \sigma_{\text{eff}}^2 I\right), \quad \sigma_{\text{eff}} = \frac{C \cdot \sigma}{|S|}$$

Where:
- $C$ is `dp_clip_norm` (default 1.0)
- $\sigma$ is `dp_noise_multiplier` (default 0.5)
- $|S|$ is the number of participating clients.

### Step 3: Rényi DP (RDP) Accounting
Tracks cumulative privacy budget spent $(\epsilon, \delta)$ across communication rounds using the moments accountant for subsampled Gaussian mechanisms:

$$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \epsilon_{\text{RDP}}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$

---

## 3. Running Simulations with Differential Privacy

Enable DP with CLI flags:
```powershell
python backend/run_simulation.py `
    --dataset mnist `
    --model cnn `
    --algorithm fedavg `
    --clients 5 `
    --rounds 10 `
    --enable-dp `
    --dp-clip-norm 1.0 `
    --dp-noise-multiplier 0.5 `
    --dp-target-delta 1e-5
```

The terminal and `metrics.json` will report privacy expenditure after each round:
```text
Round 1/10
Selected Clients : 5/5
Global Loss      : 0.1245
Global Accuracy  : 96.42%
Privacy Spent    : eps = 0.5421 (delta = 1e-05)
```

---

## 4. Gradient Inversion Defense Validation
The `InversionAttackSimulator` in [backend/fl_engine/privacy/inversion_defense.py](file:///f:/PS32/backend/fl_engine/privacy/inversion_defense.py) verifies empirical defense:
- **Without DP**: Attackers match gradients to reconstruct inputs with low Mean Squared Error (MSE).
- **With DP**: Injected noise disrupts gradient descent matching, preventing image or feature reconstruction and causing MSE to remain high.
