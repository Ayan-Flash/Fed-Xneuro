# Fed-XNeuro: Comprehensive Implementation & Engineering Summary

**Author / Maintainer:** Somya  
**Project:** Fed-XNeuro (Explainable Multimodal Federated Learning for Privacy-Preserving Mild Cognitive Impairment to Alzheimer's Disease Progression Prediction)  
**Repository Root:** `d:\Fed-Xneuro`  
**Date:** September 2026  

---

## 1. Executive Summary

Fed-XNeuro is a decentralized clinical artificial intelligence platform designed to predict the progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) while guaranteeing patient privacy across participating healthcare networks (e.g., Hospital A, Hospital B, Hospital C).

This document details all technical systems, neural architectures, privacy layers, explainability engines, backend web services, database persistence mechanisms, real-time WebSocket pipelines, and testing suites implemented across the codebase.

---

## 2. Core Fed-XNeuro Algorithmic Architecture

Implemented according to [`docs/Fed-XNeuro_Architecture_and_Algorithm.md`](docs/Fed-XNeuro_Architecture_and_Algorithm.md).

### 2.1 Multimodal Neural Network (`backend/fl_engine/models/fedxneuro.py`)
The `FedXNeuroModel` integrates three clinical modalities over longitudinal patient visit timelines:

1. **Module 1: 3D ResNet MRI Encoder (`ResNet3DEncoder`)**
   - Extracts 3D volumetric spatial features from structural T1-weighted brain MRI volumes $(B, 1, D, H, W)$.
   - Incorporates 3D convolutional residual blocks (`ResNet3DBlock`) with batch normalization, ReLU activation, and adaptive 3D average pooling, yielding a compact 128-dimensional neuroimaging embedding per visit.

2. **Module 2: Multimodal Feature Fusion (`MultimodalFusion`)**
   - Jointly aligns and projects heterogeneous feature spaces into a shared representation:
     - **3D MRI Scans**: 128-dim neuroimaging embedding.
     - **Cognitive Assessments**: 5-dim vector (MMSE, CDR-SB, ADAS-Cog11, ADAS-Cog13, FAQ).
     - **EHR Clinical Variables**: 10-dim vector (Age, Sex, Education, APOE4 allele count, BMI, Systolic BP, Diastolic BP, Total Cholesterol, Blood Glucose, Smoking status).
   - Projects concatenated modalities through linear layers, LayerNorm, and Dropout into a unified 128-dim visit representation vector $h_t$.

3. **Module 3: Missing-Visit Imputation Module (`MissingVisitImputationModule`)**
   - Clinical longitudinal studies suffer from missed clinic visits and irregular follow-up intervals ($\Delta t \in \{6\text{M}, 12\text{M}, 24\text{M}, 36\text{M}, 48\text{M}\}$).
   - Injects continuous sinusoidal temporal encodings $\text{PE}(\Delta t)$ to handle irregular temporal spacing.
   - Utilizes learnable missing-visit mask tokens and multi-head self-attention (`nn.TransformerEncoderLayer`) to impute unobserved visit representations from surrounding observed visits.

4. **Module 4: Longitudinal Temporal Transformer (`LongitudinalTransformer`)**
   - Models progression trajectories across the longitudinal visit sequence with bidirectional temporal self-attention.
   - Aggregates sequence dynamics into a unified patient representation using temporal attention pooling.

5. **Risk Prediction Head (`RiskPredictionHead`)**
   - Multilayer perceptron that maps the temporal disease state to the probability of MCI-to-AD progression:
     $$P(\text{Progression} \mid \text{Visits}_{0:T}) = \sigma(W_r \cdot z + b_r) \in [0, 1]$$
   - Categorizes risk into clinical strata:
     - **LOW**: $P < 30\%$
     - **MODERATE**: $30\% \le P \le 70\%$
     - **HIGH**: $P > 70\%$

Registered in `ModelRegistry` under `"fedxneuro"` and `"fed_xneuro"`.

---

### 2.2 Differential Privacy Guard (`backend/fl_engine/algorithms/proposed/privacy.py`)

Implements Modules 6 & 10 of the architecture specification:
- **Parameter Update $L_2$ Norm Clipping**:
  $$\Delta w_i \leftarrow \Delta w_i \cdot \min\left(1, \frac{C_{clip}}{\|\Delta w_i\|_2}\right)$$
- **Calibrated Gaussian Noise Injection**:
  $$\widetilde{\Delta w}_i = \Delta w_i + \mathcal{N}\left(0, \frac{\sigma^2 C_{clip}^2}{D_{total}} I\right)$$
  Calibrated such that total added noise across parameter dimensions maintains target variance $\sigma^2 C^2$.
- **Privacy Budget Accounting**:
  Tracks $(\epsilon, \delta)$-Differential Privacy consumption across communication rounds using RDP/Moments Accountant composition:
  $$\epsilon = \frac{q \cdot \sqrt{2 R \ln(1/\delta)}}{\sigma}$$
- **Zero Raw Data Transmission**: Guarantees that no raw MRI scans, clinical measurements, or patient identifiers leave the local hospital firewall.

---

### 2.3 Local Explainability Engine (`backend/fl_engine/algorithms/proposed/explainability.py`)

Implements Module 5 on-device clinician explainability:
- **3D Integrated Gradients for Neuroimaging (`IntegratedGradientsMRI`)**:
  - Approximates path integrals between a zero-baseline MRI volume $x'$ and patient scan $x$:
    $$\text{Attr}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$
  - Identifies critical neuroanatomical ROIs, localizing focal medial temporal lobe and hippocampal atrophy.
- **Clinical SHAP Feature Attributor (`ClinicalSHAPAttributor`)**:
  - Evaluates feature attributions across clinical and cognitive scores (MMSE, CDR-SB, APOE4, Age, Blood Pressure).
  - Ranks risk-driving clinical factors for clinicians.

---

### 2.4 Federated Optimization Algorithms (`backend/fl_engine/algorithms/proposed/`)

1. **Proposed Algorithm 1: `FedXNeuro` (`fedxneuro.py`, `algorithm_1.py`)**
   - Orchestrates multi-center federated training across hospital nodes.
   - `FedXNeuroClient`: Encapsulates local hospital execution, private patient cohort dataloaders, on-device training, local DP noise injection, and on-device XAI generation.
   - `FedXNeuroTrainer`: Multimodal longitudinal trainer with `BCEWithLogitsLoss`, gradient clipping, and numerical stability bounds.
   - `FedXNeuro`: Server-side aggregation with sample-weighted averaging and DP tracking. Registered in algorithm registry.

2. **Proposed Algorithm 2: `FedXNeuroPersonalized` (`algorithm_2.py`)**
   - Supports hospital-specific personalization under non-IID patient population distributions.
   - Federates and aggregates the shared multimodal representation backbone (`encoder_3d`, `fusion`, `imputer`, `transformer`) across all centers, while preserving center-specific personalized risk classification heads (`risk_head`) tuned to local patient demographics.
   - Registered in `ALGORITHM_REGISTRY`.

3. **FedProx Proximal Loss Regularization (`backend/fl_engine/core/trainer.py`)**
   - Implemented the proximal regularization penalty in `Trainer.train`:
     $$\mathcal{L}_{prox}(w) = \mathcal{L}_{task}(w) + \frac{\mu}{2} \|w - w_{global}\|^2$$
   - Prevents local client drift under severe non-IID statistical heterogeneity.
   - Unskipped and validated `backend/tests/unit/test_fedprox.py`.

---

### 2.5 Multimodal Dataset & Zero-Leakage Splitting (`backend/fl_engine/datasets/multimodal.py`)

- `ADNIStyleMultimodalDataset`: Synthesizes realistic ADNI-standard longitudinal patient cohorts:
  - 3D MRI structural volumes ($1 \times 16 \times 16 \times 16$).
  - Longitudinal visits with timestamps ($t \in \{0, 6, 12, 24, 36, 48\}$ months).
  - Random missing-visit masks mimicking clinical trial dropouts.
  - Cognitive batteries and EHR panels correlated with true progression targets.
- **Patient-Level Splitter**: Splits cohorts at the unique patient identifier level (`patient_id`), guaranteeing zero longitudinal cross-visit data leakage between hospital clients and evaluation sets.
- Registered under `"multimodal"` and `"adni_mci"` in `DatasetRegistry`.

---

### 2.6 Clinician Dashboard & Simulation CLI (`dashboard.py`, `run_fedxneuro.py`)

- `ClinicianDashboard`: Generates multi-format explainability outputs:
  - Terminal ASCII dashboard card with progress bars.
  - JSON clinical reports containing structured patient metrics, risk classification, and SHAP feature importances.
  - Standalone interactive HTML report with visual scorecards.
- `backend/run_fedxneuro.py`: CLI tool for multi-hospital simulations with flags for `--clients`, `--rounds`, `--patients`, `--dp`, `--clip-norm`, `--noise-multiplier`, and `--explain`.
- `backend/fl_engine/algorithms/proposed/utils.py`: Pure NumPy & PyTorch implementations of clinical metrics (Accuracy, Sensitivity, Specificity, F1 Score, ROC-AUC, PR-AUC, Brier Score) with zero dependency on NumPy 1.x / SciPy C-extensions.

---

## 3. Web Application & API Infrastructure (`backend/app/`)

Implemented all previously empty/stubbed modules into a production-grade FastAPI web application:

```
backend/app/
├── api/
│   ├── deps.py                     # Auth & SQLAlchemy DB session dependencies
│   └── routes/
│       ├── auth.py                 # POST /login, POST /register, GET /me
│       ├── users.py                # User account CRUD
│       ├── simulations.py          # Create, list, retrieve, update, delete simulations
│       ├── algorithms.py           # Enumerate FL algorithms (FedAvg, FedProx, FedXNeuro, FedXNeuroPersonalized)
│       ├── models.py               # Enumerate neural models (CNN, MLP, FedXNeuro)
│       ├── datasets.py             # Enumerate datasets (MNIST, CIFAR-10, Multimodal ADNI)
│       ├── metrics.py              # Round-by-round metric querying
│       ├── results.py              # Simulation results & GET /{id}/clinician-dashboard
│       ├── clients.py              # Connected FL client telemetry
│       └── experiments.py          # Experiment tracking
├── core/
│   ├── config.py                   # Pydantic BaseSettings (DB URL, JWT secrets, CORS)
│   ├── security.py                 # Bcrypt password hashing & JWT token handling
│   ├── exceptions.py               # Custom HTTP exception hierarchy
│   └── logging.py                  # Structured application logging
├── db/
│   ├── session.py                  # SQLite engine, SessionLocal, declarative Base
│   ├── database.py                 # init_db schema initialization
│   └── models/
│       ├── simulation.py           # Simulation ORM model
│       ├── metric.py               # Per-round Metric ORM model
│       ├── user.py                 # User ORM model
│       ├── experiment.py           # Experiment ORM model
│       ├── client.py               # Client ORM model
│       ├── result.py               # Result ORM model
│       ├── algorithm.py            # Algorithm ORM model
│       ├── model.py                # Model ORM model
│       ├── dataset.py              # Dataset ORM model
│       └── project.py              # Project ORM model
├── schemas/                        # Pydantic v2 schemas for all entities
├── services/
│   ├── simulation_service.py       # Simulation lifecycle & cascade deletion
│   ├── metrics_service.py          # Round metric persistence & querying
│   ├── algorithm_service.py        # Dynamic registry discovery
│   ├── model_service.py            # Model registry inspection
│   ├── dataset_service.py          # Dataset registry inspection
│   ├── auth_service.py             # User authentication logic
│   └── result_service.py           # Results aggregation
├── websocket/
│   ├── manager.py                  # ConnectionManager for active client WebSockets
│   └── simulation_socket.py        # /ws/simulations/{id} real-time progress broadcast
├── workers/
│   ├── simulation_worker.py        # SimulationWorker managing background runs
│   └── tasks.py                    # Celery / background task adapters
└── main.py                         # FastAPI app factory, CORS, and lifespan init
```

---

## 4. DevOps, Tooling & Root Configuration

- **[`docker-compose.yml`](docker-compose.yml)**: Multi-container configuration for backend application and database deployment.
- **[`.env.example`](.env.example)**: Comprehensive configuration template covering JWT tokens, database paths, and default hyperparameters.
- **[`Makefile`](Makefile)**: Developer shortcuts for `make install`, `make test`, `make run`, and `make clean`.
- **[`pytest.ini`](pytest.ini)**: PyTest configuration specifying test paths, auto-asyncio (`asyncio_mode = auto`), and warning filters.
- **[`LICENSE`](LICENSE)**: MIT Open Source License.

---

## 5. Verification & Test Suite Summary

Executed the complete automated test suite (`python -m pytest backend/tests/ -v`).  
**Result: 34 tests passed, 0 failed, 0 skipped.**

### 5.1 Unit Tests (27 Passed)
| Test Suite | Tests Passed | Validated Behavior |
| :--- | :---: | :--- |
| `test_fedxneuro.py` | 9 | 3D ResNet encoder, multimodal fusion, missing-visit imputation, temporal transformer, end-to-end model, DP guard, explainability engine, patient partitioning, dashboard rendering |
| `test_fedprox.py` | 2 | FedProx algorithm initialization & proximal regularization loss penalty |
| `test_fedavg.py` | 5 | Weighted parameter averaging, algorithm wrapper, integer tensor handling, error checks |
| `test_client.py` | 3 | Local client training parameter updates, evaluation, parameter get/set |
| `test_server.py` | 2 | Client selection logic and federated round coordination |
| `test_partitioning.py` | 5 | IID, Dirichlet non-IID, class skew, and unbalanced data partitioning |
| `test_simulation.py` | 1 | SimulationEngine end-to-end multi-round execution |

### 5.2 Integration Tests (7 Passed)
| Test Suite | Tests Passed | Validated Behavior |
| :--- | :---: | :--- |
| `test_api.py` | 5 | `/` root, `/health`, `/api/v1/algorithms`, `/api/v1/models`, `/api/v1/datasets`, simulation CRUD lifecycle, clinician dashboard endpoint |
| `test_fedxneuro_pipeline.py` | 1 | End-to-end multi-hospital Fed-XNeuro federated training, DP clipping, and evaluation |
| `test_simulation.py` | 1 | SimulationService lifecycle, database persistence, and metric recording |

---

## 6. Zero Empty Files Verification

A recursive audit of the workspace confirmed that **no empty or 0-byte source code files remain** in the project. The only 0-byte files in the repository are standard Python package markers:
- `backend/tests/__init__.py`
- `backend/tests/integration/__init__.py`
- `backend/tests/unit/__init__.py`
