"""
Multimodal Longitudinal Dataset for Fed-XNeuro.

Implements Module 1 & 2 data pipeline and Section 11 patient-level partitioning:
- Simulates realistic longitudinal cohorts (ADNI-style) with:
  - 3D structural MRI sequences (with simulated hippocampal atrophy & ventricular enlargement)
  - Longitudinal Cognitive scores (MMSE, CDR-SB, ADAS-Cog11, ADAS-Cog13, FAQ)
  - Longitudinal EHR records (Age, APOE4 status, Sex, Education, Systolic BP, BMI, Cholesterol)
  - Irregular visit time gaps and realistic visit dropouts (visit masks)
  - Binary progression target: MCI -> AD conversion
- Patient-level splitting ensuring ZERO longitudinal data leakage across visits.
"""

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from backend.fl_engine.datasets.base import BaseDatasetManager


class ADNIStyleMultimodalDataset(Dataset):
    """
    Realistic longitudinal multimodal dataset mimicking ADNI MCI cohorts.
    Each item represents a single patient's complete longitudinal history.
    """

    def __init__(
        self,
        num_patients: int = 300,
        num_visits: int = 6,
        mri_shape: Tuple[int, int, int] = (8, 16, 16),
        progression_rate: float = 0.35,
        missing_rate: float = 0.20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_patients = num_patients
        self.num_visits = num_visits
        self.mri_shape = mri_shape
        self.progression_rate = progression_rate
        self.missing_rate = missing_rate
        self.seed = seed

        self.patient_ids: List[str] = []
        self.labels: List[int] = []
        self.mri_data: List[torch.Tensor] = []
        self.cog_data: List[torch.Tensor] = []
        self.ehr_data: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.time_gaps: List[torch.Tensor] = []

        self._generate_cohort()

    def _generate_cohort(self) -> None:
        rng = np.random.RandomState(self.seed)
        nominal_gaps = np.array([0.0, 6.0, 12.0, 24.0, 36.0, 48.0])[: self.num_visits]
        d, h, w = self.mri_shape

        for i in range(self.num_patients):
            pid = f"PAT_{i+1:04d}"
            self.patient_ids.append(pid)

            # Assign conversion label (1 = progressor to AD, 0 = stable MCI)
            is_converter = 1 if rng.rand() < self.progression_rate else 0
            self.labels.append(is_converter)

            # 1. EHR Features
            age_base = rng.normal(73.0, 6.0)
            apoe4_prob = [0.35, 0.45, 0.20] if is_converter else [0.65, 0.30, 0.05]
            apoe4 = int(rng.choice([0, 1, 2], p=apoe4_prob))
            sex = int(rng.choice([0, 1]))  # 0: male, 1: female
            education = max(8.0, rng.normal(15.0, 2.5))
            systolic_bp = rng.normal(132.0, 14.0)
            bmi = rng.normal(26.5, 3.5)
            cholesterol = rng.normal(210.0, 25.0)

            # 2. Time gaps with clinical schedule jitter
            jitter = rng.uniform(-0.5, 0.8, size=self.num_visits)
            jitter[0] = 0.0  # Baseline has 0 gap
            patient_gaps = np.maximum(0.0, nominal_gaps + jitter)

            # 3. Longitudinal Cognitive Scores
            # MMSE: baseline ~27; progressors decline to ~18; stable ~26-27
            # CDR-SB: baseline ~1.5; progressors increase to ~6.0; stable ~1.5
            mmse_base = rng.normal(27.2, 1.5)
            cdrsb_base = max(0.5, rng.normal(1.6, 0.6))
            adas11_base = rng.normal(11.5, 3.0)
            adas13_base = rng.normal(18.0, 4.0)
            faq_base = max(0.0, rng.normal(2.5, 1.5))

            cog_seq = []
            ehr_seq = []
            mri_seq = []
            mask_seq = []

            for v in range(self.num_visits):
                t_ratio = float(v) / max(1, self.num_visits - 1)
                # Missing visit simulation (baseline visit 0 is always observed)
                if v == 0:
                    is_observed = 1.0
                else:
                    is_observed = 1.0 if rng.rand() >= self.missing_rate else 0.0
                mask_seq.append(is_observed)

                # Cognitive trajectory
                if is_converter:
                    mmse = max(10.0, mmse_base - (8.5 * (t_ratio ** 1.3)) + rng.normal(0, 0.5))
                    cdrsb = min(18.0, cdrsb_base + (5.5 * (t_ratio ** 1.3)) + rng.normal(0, 0.3))
                    adas11 = min(70.0, adas11_base + (18.0 * t_ratio) + rng.normal(0, 1.0))
                    adas13 = min(85.0, adas13_base + (25.0 * t_ratio) + rng.normal(0, 1.2))
                    faq = min(30.0, faq_base + (12.0 * t_ratio) + rng.normal(0, 0.8))
                else:
                    mmse = max(22.0, mmse_base - (1.0 * t_ratio) + rng.normal(0, 0.5))
                    cdrsb = min(3.5, cdrsb_base + (0.5 * t_ratio) + rng.normal(0, 0.3))
                    adas11 = adas11_base + (2.0 * t_ratio) + rng.normal(0, 0.8)
                    adas13 = adas13_base + (3.0 * t_ratio) + rng.normal(0, 1.0)
                    faq = min(6.0, faq_base + (1.0 * t_ratio) + rng.normal(0, 0.5))

                # Normalize cognitive scores to roughly [0, 1] range
                cog_v = [
                    (30.0 - mmse) / 20.0,   # Inverse MMSE: higher = worse
                    cdrsb / 18.0,           # CDR-SB normalized
                    adas11 / 70.0,          # ADAS-11 normalized
                    adas13 / 85.0,          # ADAS-13 normalized
                    faq / 30.0,             # FAQ normalized
                ]
                cog_seq.append(cog_v)

                # EHR features across visits
                age_v = (age_base + (patient_gaps[v] / 12.0) - 70.0) / 15.0
                ehr_v = [
                    age_v,
                    apoe4 / 2.0,
                    float(sex),
                    (education - 12.0) / 6.0,
                    (systolic_bp - 130.0) / 20.0,
                    (bmi - 25.0) / 5.0,
                    (cholesterol - 200.0) / 40.0,
                ]
                ehr_seq.append(ehr_v)

                # Structural 3D MRI generation:
                # Simulates brain parenchyma with central ventricles and medial temporal lobes
                vol = np.zeros((1, d, h, w), dtype=np.float32)

                # Ellipsoidal brain background
                zz, yy, xx = np.ogrid[:d, :h, :w]
                zc, yc, xc = d / 2.0, h / 2.0, w / 2.0
                rad_sq = ((zz - zc) / (d * 0.45)) ** 2 + ((yy - yc) / (h * 0.45)) ** 2 + ((xx - xc) / (w * 0.45)) ** 2
                brain_mask = rad_sq <= 1.0
                vol[0][brain_mask] = 0.75 + rng.normal(0, 0.05, size=brain_mask.sum())

                # Ventricles (hypointense CSF center, expands with AD progression)
                ventricle_scale = (1.0 + (1.5 * t_ratio if is_converter else 0.2 * t_ratio))
                v_rad_sq = ((zz - zc) / (d * 0.15 * ventricle_scale)) ** 2 + ((yy - yc) / (h * 0.20 * ventricle_scale)) ** 2 + ((xx - xc) / (w * 0.15 * ventricle_scale)) ** 2
                ventricle_mask = v_rad_sq <= 1.0
                vol[0][ventricle_mask] = 0.15 + rng.normal(0, 0.02, size=ventricle_mask.sum())

                # Bilateral hippocampus (medial temporal region, atrophies with AD)
                hippo_intensity = 0.85 - (0.40 * t_ratio if is_converter else 0.05 * t_ratio)
                vol[0, int(zc), int(yc), int(xc - w * 0.25)] = hippo_intensity
                vol[0, int(zc), int(yc), int(xc + w * 0.25)] = hippo_intensity

                # Intensity normalization to [0, 1]
                vol = np.clip(vol, 0.0, 1.0)
                mri_seq.append(vol)

            self.mri_data.append(torch.tensor(np.stack(mri_seq, axis=0), dtype=torch.float32))  # [T, 1, D, H, W]
            self.cog_data.append(torch.tensor(np.array(cog_seq), dtype=torch.float32))          # [T, 5]
            self.ehr_data.append(torch.tensor(np.array(ehr_seq), dtype=torch.float32))          # [T, 7]
            self.masks.append(torch.tensor(np.array(mask_seq), dtype=torch.float32))            # [T]
            self.time_gaps.append(torch.tensor(patient_gaps, dtype=torch.float32))              # [T]

    def __len__(self) -> int:
        return self.num_patients

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "mri": self.mri_data[idx],
            "cognitive": self.cog_data[idx],
            "ehr": self.ehr_data[idx],
            "visit_mask": self.masks[idx],
            "time_gaps": self.time_gaps[idx],
            "label": torch.tensor([float(self.labels[idx])], dtype=torch.float32),
            "patient_id": self.patient_ids[idx],
        }

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels, dtype=np.int64)


class MultimodalDatasetManager(BaseDatasetManager):
    """
    Manages longitudinal multimodal datasets and enforces patient-level partitioning.
    No patient's visits ever cross between train/test or across hospital clients.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/multimodal",
        num_patients: int = 300,
        num_visits: int = 6,
        mri_shape: Tuple[int, int, int] = (8, 16, 16),
        seed: int = 42,
    ) -> None:
        super().__init__(data_dir=data_dir)
        self.num_patients = num_patients
        self.num_visits = num_visits
        self.mri_shape = mri_shape
        self.seed = seed
        self.dataset: Optional[ADNIStyleMultimodalDataset] = None
        self.train_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Creates cohort and performs patient-level train/test partition (80/20).
        """
        self.dataset = ADNIStyleMultimodalDataset(
            num_patients=self.num_patients,
            num_visits=self.num_visits,
            mri_shape=self.mri_shape,
            seed=self.seed,
        )

        n_total = len(self.dataset)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n_total)

        n_train = int(n_total * 0.80)
        train_idx = indices[:n_train].tolist()
        test_idx = indices[n_train:].tolist()

        self.train_dataset = Subset(self.dataset, train_idx)
        self.test_dataset = Subset(self.dataset, test_idx)
        return self.train_dataset, self.test_dataset

    def get_targets(self) -> np.ndarray:
        if self.train_dataset is None:
            self.load_data()
        all_labels = self.dataset.get_labels()
        train_indices = self.train_dataset.indices
        return all_labels[train_indices]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "adni_mci_multimodal",
            "num_classes": 2,
            "classes": ["Stable MCI", "AD Progression"],
            "modalities": ["3D MRI", "Cognitive Scores", "Longitudinal EHR"],
            "visits_per_patient": self.num_visits,
            "mri_shape": list(self.mri_shape),
        }
