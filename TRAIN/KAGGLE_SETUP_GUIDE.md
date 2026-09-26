# 🧠 Fed-XNeuro: Kaggle Training Setup Guide

This guide explains **exactly** how to run the Fed-XNeuro training notebook on Kaggle, including how to upload and use the ADNI/OASIS datasets (which are not hosted on Kaggle).

---

## 📋 Quick Start (No Real Data Needed)

The notebook works **out of the box** with synthetic data. If you just want to test the pipeline:

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**
2. Copy-paste the contents of `kaggle_train_fedxneuro.py` into the notebook
3. Under **Settings** (right panel) → **Accelerator** → Select **GPU T4 x2** or **GPU P100**
4. Click **"Run All"** — it will auto-generate a 300-patient synthetic cohort and begin training

---

## 🗂️ Option A: Upload ADNI Dataset to Kaggle

Since ADNI is **not publicly hosted on Kaggle**, you need to create a private Kaggle Dataset:

### Step 1: Download from ADNI

1. Go to [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
2. Create a LONI IDA account → Submit a Data Use Agreement (DUA)
3. Once approved, download:
   - **`ADNIMERGE.csv`** (from "Data & Samples" → "Study Data" → "ADNIMERGE")
   - **MRI Scans** (optional): Under "Image Collections" → Search → Modality: `MRI`, Weighting: `T1`, Format: `NIfTI`

### Step 2: Create a Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **"New Dataset"**
2. Name it something like `adni-mci-cohort`
3. **Set visibility to PRIVATE** (ADNI's DUA prohibits public redistribution)
4. Upload your files in this structure:
   ```
   adni-mci-cohort/
   ├── ADNIMERGE.csv
   └── mri/                     ← (optional) NIfTI MRI files
       ├── 002_S_0295_MR_T1.nii
       └── ...
   ```
5. Click **"Create"**

### Step 3: Attach to Your Notebook

1. Open your notebook on Kaggle
2. Right panel → **"Add Data"** → Search for your private dataset `adni-mci-cohort`
3. Click **"Add"** — it will mount at `/kaggle/input/adni-mci-cohort/`

### Step 4: Configure the Notebook

In the `CFG = TrainingConfig(...)` section of the notebook, set:

```python
CFG = TrainingConfig(
    dataset="adni",
    data_dir="/kaggle/input/adni-mci-cohort",  # ← your dataset slug
    adni_csv="ADNIMERGE.csv",
    mri_subfolder="mri",
    # ... rest of config
)
```

The notebook auto-searches all subdirectories under `data_dir` for the CSV file, so minor directory nesting differences are handled automatically.

---

## 🗂️ Option B: Upload OASIS Dataset to Kaggle

### Step 1: Download from OASIS

1. Go to [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
2. Navigate to **OASIS-2: Longitudinal MRI** (open access)
3. Download:
   - **`oasis_longitudinal.csv`**
   - **MRI scans** (NIfTI format, optional)

### Step 2: Create a Kaggle Dataset

Same process as ADNI above. Name it `oasis-longitudinal`:
```
oasis-longitudinal/
├── oasis_longitudinal.csv
└── mri/                         ← (optional)
    ├── OAS2_0001_MR1.nii
    └── ...
```

> **Note:** OASIS-2 is open-access, so you can set this dataset to **public** if desired.

### Step 3: Configure

```python
CFG = TrainingConfig(
    dataset="oasis",
    data_dir="/kaggle/input/oasis-longitudinal",
    oasis_csv="oasis_longitudinal.csv",
    mri_subfolder="mri",
    # ...
)
```

---

## 🗂️ Option C: Upload via Kaggle API (CLI)

If you prefer the command line:

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your API token at ~/.kaggle/kaggle.json
#    (Download from kaggle.com → Account → API → Create New Token)

# 3. Create a dataset metadata file
mkdir adni-mci-cohort
cp ADNIMERGE.csv adni-mci-cohort/
# (optional) cp -r mri/ adni-mci-cohort/mri/

# 4. Initialize dataset metadata
kaggle datasets init -p ./adni-mci-cohort

# 5. Edit dataset-metadata.json:
#    - Set "title": "adni-mci-cohort"
#    - Set "id": "your-username/adni-mci-cohort"

# 6. Upload (private by default)
kaggle datasets create -p ./adni-mci-cohort --dir-mode zip

# 7. To update an existing dataset:
kaggle datasets version -p ./adni-mci-cohort -m "Updated CSV"
```

---

## ⚙️ Kaggle Notebook Settings

### Required Settings

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 (recommended) or GPU P100 |
| **Language** | Python |
| **Internet** | ON (for pip installs in first cell) |
| **Persistence** | Files saved to `/kaggle/working/` |

### Enable GPU

Right panel → **Settings** → **Accelerator** → Select **GPU T4 x2**

> **Without GPU**, training will still work but will be significantly slower (~10x). The notebook auto-detects `cuda` vs `cpu`.

### Kaggle GPU Quota

- **Free accounts**: 30 hours/week of GPU
- **Phone-verified**: Required to access GPUs
- Go to **kaggle.com → Account → Phone Verification** to enable

---

## 🏋️ Training Configurations

### Quick Test Run (Synthetic Data, ~5 min)
```python
CFG = TrainingConfig(
    dataset="synthetic",
    algorithm="centralized",
    num_rounds=5,
    num_synthetic_patients=100,
    batch_size=16,
    mri_shape=(8, 8, 8),
)
```

### Standard Federated Training (ADNI, ~20 min with GPU)
```python
CFG = TrainingConfig(
    dataset="adni",
    data_dir="/kaggle/input/adni-mci-cohort",
    algorithm="fedavg",
    num_clients=5,
    num_rounds=10,
    local_epochs=3,
    batch_size=8,
    mri_shape=(16, 16, 16),
)
```

### FedProx with Non-IID Robustness
```python
CFG = TrainingConfig(
    dataset="adni",
    data_dir="/kaggle/input/adni-mci-cohort",
    algorithm="fedprox",
    mu=0.01,
    num_clients=5,
    num_rounds=15,
    local_epochs=3,
)
```

### High-Resolution Full Training (~2-4 hours with GPU)
```python
CFG = TrainingConfig(
    dataset="adni",
    data_dir="/kaggle/input/adni-mci-cohort",
    algorithm="fedavg",
    num_clients=5,
    num_rounds=30,
    local_epochs=5,
    mri_shape=(32, 32, 32),
    fusion_dim=256,
    learning_rate=5e-4,
    batch_size=4,
)
```

### Differential Privacy Training
```python
CFG = TrainingConfig(
    dataset="adni",
    data_dir="/kaggle/input/adni-mci-cohort",
    algorithm="fedavg",
    enable_dp=True,
    dp_clip_norm=1.0,
    dp_noise_multiplier=0.5,
    num_rounds=10,
)
```

---

## 📤 Downloading Results from Kaggle

All outputs are saved to `/kaggle/working/`:

| File | Description |
|------|-------------|
| `fedxneuro_best.pt` | Best model checkpoint (by AUC-ROC) |
| `training_history.json` | Per-round metrics (loss, accuracy, AUC) |
| `training_curves.png` | Visualization of training curves |

### Download from UI
1. After notebook finishes → Click **"Output"** tab
2. Click **"Download All"** to get a zip of all output files

### Download via API
```bash
kaggle kernels output your-username/your-notebook-slug -p ./results
```

### Load Checkpoint Locally
```python
import torch

checkpoint = torch.load("fedxneuro_best.pt", map_location="cpu")
print(f"Best AUC: {checkpoint['best_auc']:.4f}")
print(f"Config: {checkpoint['config']}")

# Reload into model
from TRAIN.kaggle_train_fedxneuro import FedXNeuroModel, TrainingConfig
cfg = TrainingConfig(**checkpoint['config'])
model = FedXNeuroModel(cfg)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🔄 Converting .py to .ipynb (Optional)

The `.py` file uses `# %%` cell markers that Kaggle, VS Code, and Jupyter all understand.
To convert to a proper `.ipynb`:

```bash
pip install jupytext
jupytext --to notebook TRAIN/kaggle_train_fedxneuro.py
# Creates: TRAIN/kaggle_train_fedxneuro.ipynb
```

Or simply copy-paste the entire `.py` file contents into a single Kaggle notebook cell — the `# %%` markers will auto-create separate cells.

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ADNIMERGE.csv not found` | Check your dataset is attached and the `data_dir` path matches the Kaggle slug |
| `CUDA out of memory` | Reduce `batch_size` to 2-4, or reduce `mri_shape` to `(8,8,8)` |
| `nibabel ImportError` | Internet must be ON for the first cell to run `pip install` |
| Training is slow on CPU | Enable GPU in Settings → Accelerator |
| `ModuleNotFoundError` | The notebook is self-contained. Do NOT try to import from `backend/` |
| Dataset not showing in "Add Data" | Make sure you created it under your Kaggle account |

---

## 📁 File Inventory

| File | Purpose |
|------|---------|
| [`kaggle_train_fedxneuro.py`](kaggle_train_fedxneuro.py) | Self-contained training notebook (copy to Kaggle) |
| [`KAGGLE_SETUP_GUIDE.md`](KAGGLE_SETUP_GUIDE.md) | This guide |
| [`DATASETS_GUIDE.md`](DATASETS_GUIDE.md) | Detailed ADNI/OASIS data acquisition guide |
