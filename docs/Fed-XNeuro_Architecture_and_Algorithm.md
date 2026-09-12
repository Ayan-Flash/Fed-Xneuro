# Fed-XNeuro: Explainable Multimodal Federated Learning for Privacy-Preserving MCI to AD Progression Prediction

## 1. Project Overview

**Fed-XNeuro** is an Explainable Multimodal Federated Learning framework
for privacy-preserving, longitudinal prediction of progression from Mild
Cognitive Impairment (MCI) to Alzheimer's Disease (AD).

The system combines: - MRI sequences - Cognitive scores - Longitudinal
EHR records - Multimodal feature fusion - Missing-visit handling -
Longitudinal Transformer modeling - Explainable AI - Differential
privacy - Federated learning with FedAvg - A clinician-facing dashboard

This document records the modified Fed-XNeuro algorithm and complete
system architecture from the supplied project architecture document.

------------------------------------------------------------------------

# 2. Fed-XNeuro Algorithm

## Algorithm: Fed-XNeuro

### Input

-   MRI Sequences `M`
-   Cognitive Scores `C`
-   Longitudinal EHR Records `E`

### Output

-   MCI → AD Conversion Risk
-   Explainability Report
-   Brain Region Attribution Map

------------------------------------------------------------------------

## 3. Algorithm Steps

### Step 1 --- Initialize Global Model

Initialize the global Transformer model:

``` text
G = Global Transformer Model
```

------------------------------------------------------------------------

### Step 2 --- Initialize Federated Clients

For each federated client `Cᵢ`:

``` text
Load MRI Visits
Load Cognitive Scores
Load EHR Timeline
```

Each client represents a local data environment such as a hospital or
simulated healthcare site.

------------------------------------------------------------------------

### Step 3 --- Data Preprocessing

#### MRI preprocessing

``` text
MRI
 ↓
Skull Stripping
 ↓
Intensity Normalization
```

#### Cognitive preprocessing

``` text
Cognitive Scores
 ↓
Missing Value Imputation
```

#### EHR preprocessing

``` text
EHR Timeline
 ↓
Temporal Encoding
```

------------------------------------------------------------------------

### Step 4 --- MRI Feature Extraction

Use a 3D ResNet encoder to extract neuroimaging features:

``` text
Fᵢ = 3D-ResNet(MRI)
```

The resulting representation captures useful features from the MRI
scans.

------------------------------------------------------------------------

### Step 5 --- Multimodal Fusion

Combine the MRI, cognitive, and EHR representations:

``` text
Xᵢ = Concatenate(
    MRI Features,
    Cognitive Scores,
    EHR Features
)
```

This produces a unified multimodal representation.

------------------------------------------------------------------------

### Step 6 --- Missing-Visit Imputation Module

The model handles missing longitudinal visits using:

-   Visit sequence
-   Visit mask
-   Time gaps

The sequence is processed by a Longitudinal Attention Transformer
Encoder:

``` text
Input:
    Visit Sequence
    Visit Mask
    Time Gaps

        ↓

Longitudinal Attention Transformer Encoder

        ↓

Recover Missing Visits
```

This allows the framework to handle patient dropouts and irregular
follow-up schedules.

------------------------------------------------------------------------

### Step 7 --- Temporal Disease Modeling

The reconstructed longitudinal sequence is passed through a Temporal
Transformer:

``` text
Hᵢ = Temporal Transformer(Xᵢ')
```

The model learns:

-   Disease progression patterns
-   Relationships across visits
-   Irregular time intervals

------------------------------------------------------------------------

### Step 8 --- Risk Prediction

The prediction head produces the probability of MCI-to-AD conversion:

``` text
Pᵢ = Sigmoid(Linear(Hᵢ))
```

Output:

``` text
P(MCI → AD)
```

------------------------------------------------------------------------

### Step 9 --- Local Explainability

The framework generates explanations locally at each client.

#### MRI attribution

Use:

``` text
Integrated Gradients
```

to identify important regions/features in MRI.

#### Clinical attribution

Use:

``` text
SHAP
```

for clinical and EHR features.

The system generates:

-   Hippocampus heatmaps
-   Feature importance
-   Clinical/EHR contribution information

------------------------------------------------------------------------

### Step 10 --- Differential Privacy

Before sending model information to the federated server:

1.  Clip gradients.
2.  Add Gaussian noise.
3.  Produce a differentially private gradient.

Conceptually:

``` text
DP_Gradient = Gradient + Noise
```

The framework uses `(ε, δ)`-Differential Privacy.

------------------------------------------------------------------------

### Step 11 --- Federated Communication

The client sends only privacy-protected model information:

``` text
Send:
    DP_Gradient
```

The following are **never sent** to the server:

``` text
MRI
EHR
SHAP Maps
Attention Maps
```

Raw patient data therefore remains at the local client.

------------------------------------------------------------------------

### Step 12 --- Server Aggregation

The server aggregates the client updates using Federated Averaging:

``` text
G = FedAvg(DP_Gradients)
```

The aggregated global model represents information learned from
participating clients without collecting their raw patient data
centrally.

------------------------------------------------------------------------

### Step 13 --- Broadcast Updated Model

The updated global model is broadcast back to the participating clients:

``` text
Global Model
     ↓
Client A
Client B
Client C
Client D
```

------------------------------------------------------------------------

### Step 14 --- Repeat Until Convergence

Federated training repeats:

``` text
Local Training
      ↓
Privacy Protection
      ↓
Send Updates
      ↓
FedAvg
      ↓
Global Model
      ↓
Broadcast
      ↓
Local Training
```

The process continues until the selected convergence/stopping criterion
is reached.

------------------------------------------------------------------------

# 4. Final Outputs

After training/inference, Fed-XNeuro produces:

### 4.1 Conversion Risk Score

A numerical risk score representing predicted MCI-to-AD progression
risk.

### 4.2 AD Progression Probability

``` text
P(MCI → AD)
```

### 4.3 Brain Region Explanations

MRI attribution maps identify influential brain regions/features.

### 4.4 Clinical Feature Importance

SHAP-based explanations identify influential clinical and EHR variables.

------------------------------------------------------------------------

# 5. Complete System Architecture

``` text
                         FED-XNEURO
                              |
                              v
                     Federated Clients
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
      Hospital A          Hospital B          Hospital C
          |                   |                   |
          v                   v                   v
   MRI + Cognitive       MRI + Cognitive     MRI + Cognitive
       + EHR                 + EHR               + EHR
          |                   |                   |
          v                   v                   v
    3D ResNet Encoder   3D ResNet Encoder   3D ResNet Encoder
          |                   |                   |
          v                   v                   v
    Feature Fusion      Feature Fusion      Feature Fusion
          |                   |                   |
          v                   v                   v
 Missing Visit         Missing Visit        Missing Visit
 Transformer           Transformer          Transformer
          |                   |                   |
          v                   v                   v
 Longitudinal          Longitudinal         Longitudinal
 Transformer            Transformer          Transformer
          |                   |                   |
          v                   v                   v
 Risk Prediction       Risk Prediction      Risk Prediction
          |                   |                   |
          v                   v                   v
 SHAP + Integrated     SHAP + Integrated   SHAP + Integrated
 Gradients             Gradients            Gradients
          |                   |                   |
          v                   v                   v
 Differential          Differential        Differential
 Privacy               Privacy              Privacy
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                       Privacy-Protected
                         Model Updates
                              |
                              v
                    +----------------------+
                    |     FedAvg Server    |
                    +----------------------+
                              |
                              v
                     Updated Global Model
                              |
                              v
                    Broadcast to Clients
```

------------------------------------------------------------------------

# 6. Module Architecture

## Module 1 --- MRI Feature Extraction

**Model:** 3D ResNet18

Purpose:

-   Process 3D neuroimaging data
-   Extract meaningful neuroimaging representations
-   Produce MRI feature embeddings for multimodal fusion

``` text
T1 MRI
  ↓
3D ResNet18
  ↓
MRI Feature Embedding
```

------------------------------------------------------------------------

## Module 2 --- Multimodal Fusion

Inputs:

``` text
MRI Features
+
Cognitive Scores
+
Longitudinal EHR
```

Output:

``` text
Unified Multimodal Representation
```

------------------------------------------------------------------------

## Module 3 --- Missing-Visit Transformer

Purpose:

-   Handle patient dropouts
-   Handle missing visits
-   Apply temporal masking
-   Represent time gaps

``` text
Visit Sequence
+
Visit Mask
+
Time Gap
      ↓
Missing-Visit Transformer
      ↓
Reconstructed Sequence
```

------------------------------------------------------------------------

## Module 4 --- Longitudinal Transformer

Purpose:

-   Learn disease progression
-   Model relationships across visits
-   Handle irregular temporal intervals

``` text
Longitudinal Multimodal Sequence
              ↓
     Temporal Transformer
              ↓
       Disease State
              ↓
        Risk Prediction
```

------------------------------------------------------------------------

## Module 5 --- Explainability Layer

### MRI

``` text
Integrated Gradients
        ↓
MRI Attribution
        ↓
Brain Region Heatmap
```

### Clinical + EHR

``` text
SHAP
 ↓
Feature Importance
```

------------------------------------------------------------------------

## Module 6 --- Differential Privacy Guard

Components:

``` text
Gradient Clipping
       +
Gaussian Noise Addition
       ↓
Differentially Private Update
```

The target privacy mechanism is:

``` text
(ε, δ)-Differential Privacy
```

------------------------------------------------------------------------

## Module 7 --- Federated Aggregation

Aggregation algorithm:

``` text
FedAvg
```

Core principle:

``` text
No Raw Data Sharing
```

Clients keep their MRI, EHR, and other patient-level information
locally.

------------------------------------------------------------------------

## Module 8 --- Clinician Dashboard

The dashboard presents:

-   Risk Score
-   MRI Heatmaps
-   SHAP Feature Importance
-   Progression Timeline

Example:

``` text
+---------------------------------------+
|           FED-XNEURO DASHBOARD        |
+---------------------------------------+
|                                       |
| MCI → AD Risk                         |
|                                       |
|              78.4%                    |
|                                       |
| Risk Category: HIGH                   |
|                                       |
+---------------------------------------+
| Clinical Feature Importance           |
|                                       |
| MMSE Decline        █████████         |
| CDR                 ███████           |
| Age                 █████             |
| Other               ███               |
+---------------------------------------+
| MRI Attribution                       |
|                                       |
|          [Brain Heatmap]              |
|                                       |
+---------------------------------------+
| Longitudinal Progression              |
|                                       |
| Baseline → 6M → 12M → 24M            |
+---------------------------------------+
```

------------------------------------------------------------------------

# 7. End-to-End Data Flow

``` text
Patient Data
     |
     +---- MRI
     |
     +---- Cognitive Scores
     |
     +---- EHR
     |
     v
Preprocessing
     |
     +---- MRI normalization
     +---- Missing value imputation
     +---- Temporal encoding
     |
     v
3D ResNet MRI Encoder
     |
     v
Multimodal Fusion
     |
     v
Missing-Visit Transformer
     |
     v
Longitudinal Transformer
     |
     v
Risk Prediction Head
     |
     +------------------------+
     |                        |
     v                        v
MCI → AD Risk          Explainability
                         |
                   +-----+------+
                   |            |
                   v            v
                 SHAP      Integrated
                           Gradients
                   |            |
                   v            v
             Feature       Brain Region
             Importance      Heatmap
     |
     v
Differential Privacy
     |
     v
Federated Model Update
     |
     v
FedAvg Server
     |
     v
Global Model
     |
     v
Broadcast to Clients
     |
     v
Repeat Until Convergence
```

------------------------------------------------------------------------

# 8. Key Privacy Principle

The core privacy boundary is:

``` text
                    LOCAL CLIENT
              +----------------------+
              | MRI                  |
              | EHR                  |
              | Cognitive Data       |
              | SHAP Maps            |
              | Attention Maps       |
              +----------+-----------+
                         |
                  Local Training
                         |
                  Gradient Clipping
                         |
                  Gaussian Noise
                         |
                         v
                 DP Model Update
                         |
                         v
                    FEDAVG SERVER
```

Raw patient data stays within the local client.

------------------------------------------------------------------------

# 9. Recommended Implementation Order

To reduce project complexity, implement the system incrementally.

## Phase 1 --- Dataset and Data Pipeline

``` text
Dataset
  ↓
Patient Selection
  ↓
MCI Cohort
  ↓
Longitudinal Visits
  ↓
MCI → AD Labels
```

## Phase 2 --- Clinical Baseline

``` text
Clinical + Cognitive
        ↓
Baseline Classifier
        ↓
Evaluation
```

## Phase 3 --- MRI Encoder

``` text
MRI
 ↓
3D ResNet18
 ↓
MRI Embedding
```

## Phase 4 --- Multimodal Model

``` text
MRI
+
Clinical
+
Cognitive/EHR
 ↓
Fusion
 ↓
Prediction
```

## Phase 5 --- Longitudinal Modeling

``` text
Multiple Visits
 ↓
Missing-Visit Handling
 ↓
Temporal Transformer
 ↓
Prediction
```

## Phase 6 --- Federated Learning

``` text
Client 1
Client 2
Client 3
Client 4
 ↓
Local Training
 ↓
FedAvg
```

## Phase 7 --- Differential Privacy

``` text
Gradient
 ↓
Clipping
 ↓
Gaussian Noise
 ↓
DP Update
```

## Phase 8 --- Explainability

``` text
Clinical/EHR → SHAP
MRI → Integrated Gradients
```

## Phase 9 --- Dashboard

``` text
Risk Score
+
MRI Heatmap
+
SHAP Importance
+
Progression Timeline
```

------------------------------------------------------------------------

# 10. Important Experimental Comparisons

The final research evaluation should compare progressively more
sophisticated models.

``` text
Baseline
   ↓
Clinical-only
   ↓
MRI-only
   ↓
Clinical + Cognitive
   ↓
Centralized Multimodal
   ↓
Multimodal Transformer
   ↓
Federated Transformer
   ↓
Federated + Differential Privacy
   ↓
Fed-XNeuro
```

Potential metrics:

-   Accuracy
-   Precision
-   Recall / Sensitivity
-   Specificity
-   F1-score
-   ROC-AUC
-   PR-AUC
-   Calibration/Brier score where appropriate
-   Client-level performance
-   Communication rounds
-   Training time

------------------------------------------------------------------------

# 11. Important Experimental Considerations

### Patient-level data splitting

Training, validation, and test sets should be separated by **patient**,
not by individual visits, to avoid longitudinal data leakage.

### Federated simulation

If publicly available data do not represent actual connected hospitals,
federated clients should be described honestly as **simulated clients**
created from appropriate dataset partitions.

### Privacy claims

Federated learning should not be described as providing absolute
privacy. The project should distinguish between keeping raw data local
and additional protection provided by differential privacy.

### Explainability

SHAP and Integrated Gradients should be presented as model-attribution
methods rather than proof of causal relationships.

------------------------------------------------------------------------

# 12. Final System Goal

The completed Fed-XNeuro system should demonstrate:

``` text
Privacy
   +
Multimodal Learning
   +
Longitudinal Modeling
   +
Transformer Architecture
   +
Federated Learning
   +
Differential Privacy
   +
Explainable AI
   +
MCI → AD Prediction
```

The final objective is to predict the probability of progression from
MCI to AD while keeping patient-level data decentralized and providing
interpretable evidence for the model's prediction.
