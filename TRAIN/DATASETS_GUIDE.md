# Fed-XNeuro Datasets Guide: ADNI & OASIS

This guide details how to acquire, structure, preprocess, and ingest the **ADNI** and **OASIS** neuroimaging cohorts for training the **Fed-XNeuro** multimodal federated learning framework.

---

## 1. Overview of Cohorts

Fed-XNeuro predicts the conversion of **Mild Cognitive Impairment (MCI)** to **Alzheimer's Disease (AD)** within a 3-year prospective window using three modalities:
1. **3D Structural T1-weighted MRI** (spatial hippocampal/ventricular atrophy)
2. **Longitudinal Cognitive Assessments** (MMSE, CDR-SB, ADAS-Cog, FAQ)
3. **Electronic Health Records & Demographics** (Age, Gender, Education, APOE-ε4 allele count, volumetric biomarkers)

---

## 2. Where and How to Obtain the Datasets

### A. ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Official Portal**: [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
- **Access Steps**:
  1. Click **"Data & Samples"** &rarr; **"Access Data"**.
  2. Create a LONI IDA account and submit a Data Use Agreement (DUA) describing your research application.
  3. Approval is usually granted within 1–3 business days.
- **Files to Download**:
  1. **Clinical Spreadsheet**: Download `ADNIMERGE.csv` (the standardized master clinical table consolidating demographics, cognitive exams, diagnostic status across baseline and follow-up visits `m00` to `m36`).
  2. **MRI Scans**: Under **"Image Collections"** / **"Search"**, select:
     - Modality: `MRI`
     - Weighting: `T1` (MPRAGE or IR-FSPGR)
     - File Format: `NIfTI` (`.nii` or `.nii.gz`)

---

### B. OASIS (Open Access Series of Imaging Studies)
- **Official Portal**: [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
- **Recommended Release**: **OASIS-2 (Longitudinal MRI)** or **OASIS-3**
  - OASIS-2 is open access and directly downloadable.
- **Access Steps**:
  1. Navigate to **OASIS-2: Longitudinal MRI Data in Nondemented and Demented Older Adults**.
  2. Register for an account and agree to the open-access DUA.
- **Files to Download**:
  1. **Clinical Spreadsheet**: `oasis_longitudinal.csv` (contains `Subject ID`, `Visit`, `MR Delay`, `M/F`, `Age`, `EDUC`, `SES`, `MMSE`, `CDR`, `eTIV`, `nWBV`, `ASF`).
  2. **MRI Scans**: Structural T1-weighted NIfTI volumes for subjects across visits.

---

## 3. Directory Layout in `TRAIN/`

Place your downloaded files into the `TRAIN/data/raw/` directory following this structure:

```
TRAIN/
├── data/
│   ├── raw/
│   │   ├── adni/
│   │   │   ├── ADNIMERGE.csv               <-- Place ADNI clinical CSV here
│   │   │   └── mri/
│   │   │       ├── 002_S_0295_MR_T1.nii    <-- Place 3D NIfTI scans here
│   │   │       └── ...
│   │   └── oasis/
│   │       ├── oasis_longitudinal.csv      <-- Place OASIS clinical CSV here
│   │       └── mri/
│   │           ├── OAS2_0001_MR1.nii       <-- Place 3D NIfTI scans here
│   │           └── ...
│   └── processed/                          <-- Ingestion CLI generates cached tensors here
│       ├── adni/
│       │   ├── train_cohort.pt
│       │   ├── test_cohort.pt
│       │   └── manifest.json
│       └── oasis/
│           ├── train_cohort.pt
│           ├── test_cohort.pt
│           └── manifest.json
```

---

## 4. Preprocessing & Ingestion Pipeline

The ingestion tool performs automated verification and preprocessing:
1. **Clinical Parsing**: Extracts longitudinal visit sequences, filters MCI subjects, and tracks converter vs. non-converter status.
2. **3D MRI Spatial Resampling**: Resamples non-uniform MRI volumes into uniform isometric voxel grids (default `16×16×16` for fast FL experimentation or `32×32×32`/`64×64×64` for full training) with intensity normalization (`[0, 1]`) and foreground brain masking.
3. **Zero-Leakage Patient Splitting**: Partitions patients strictly by Subject ID (80% train, 20% test). **Zero visits from a test patient ever appear in training.**
4. **Binary Tensor Caching**: Packages datasets into `.pt` files for fast loading during federated rounds.

### Ingesting ADNI:
```bash
python TRAIN/datasets/ingest_data.py --source adni --mri-shape 16 16 16
```

### Ingesting OASIS:
```bash
python TRAIN/datasets/ingest_data.py --source oasis --mri-shape 16 16 16
```

*(Note: If raw CSVs or MRI scans are not yet placed in `TRAIN/data/raw/`, the ingestion tool automatically falls back to an integrated synthetic cohort so pipelines and tests run seamlessly!)*

---

## 5. Training with the Prepared Datasets

Once ingested or in fallback mode, launch federated training immediately:

### Train Fed-XNeuro on ADNI:
```bash
python TRAIN/training/train_fedxneuro.py --dataset adni --rounds 10 --clients 5 --algorithm fedprox --mu 0.01
```

### Train Fed-XNeuro on OASIS:
```bash
python TRAIN/training/train_fedxneuro.py --dataset oasis --rounds 10 --clients 3 --algorithm fedavg
```

### Run Differential Privacy & Gradient Compression:
```bash
python TRAIN/training/train_fedxneuro.py --dataset adni --rounds 10 --clients 5 --dp --compression topk
```

### Run the Section 10 9-Model Ablation Benchmark:
```bash
python TRAIN/training/benchmark_ablation.py --rounds 5 --clients 3
```
