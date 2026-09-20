"""
Unit tests for Real ADNI & OASIS Dataset Loaders, Preprocessing, and Parsers.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from backend.fl_engine.datasets.preprocessing import (
    resample_3d_volume,
    intensity_normalize_mri,
    extract_brain_mask,
)
from backend.fl_engine.datasets.adni_loader import (
    ADNIClinicalParser,
    ADNIMultimodalDataset,
    ADNIDatasetManager,
)
from backend.fl_engine.datasets.oasis_loader import (
    OASISClinicalParser,
    OASISMultimodalDataset,
    OASISDatasetManager,
)
from backend.fl_engine.datasets.registry import DatasetRegistry


def test_preprocessing_resample_and_normalize():
    """Verify 3D volume resampling, intensity normalization, and brain mask extraction."""
    raw_vol = np.random.rand(8, 24, 24).astype(np.float32) * 500.0
    # Add a bright sphere mimicking brain
    raw_vol[2:6, 6:18, 6:18] += 800.0

    target_shape = (16, 16, 16)
    resampled = resample_3d_volume(raw_vol, target_shape=target_shape)
    assert resampled.shape == target_shape

    norm_vol = intensity_normalize_mri(resampled, method="minmax")
    assert norm_vol.min() >= 0.0
    assert norm_vol.max() <= 1.0

    mask = extract_brain_mask(norm_vol, threshold_ratio=0.2)
    assert mask.shape == target_shape
    assert mask.dtype == bool


def test_adni_clinical_parser_and_dataset():
    """Verify parsing of sample ADNIMERGE.csv and dataset creation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "ADNIMERGE.csv")
        sample_df = pd.DataFrame({
            "PTID": ["002_S_0295", "002_S_0295", "002_S_0413", "002_S_0413"],
            "VISCODE": ["bl", "m06", "bl", "m06"],
            "Months_bl": [0.0, 6.0, 0.0, 6.0],
            "DX_bl": ["LMCI", "LMCI", "EMCI", "EMCI"],
            "DX": ["MCI", "Dementia", "MCI", "MCI"],
            "AGE": [74.5, 75.0, 68.2, 68.7],
            "PTGENDER": ["Female", "Female", "Male", "Male"],
            "PTEDUCAT": [16, 16, 14, 14],
            "APOE4": [1, 1, 0, 0],
            "MMSE": [27.0, 21.0, 28.0, 28.0],
            "CDRSB": [1.5, 5.0, 1.0, 1.0],
            "ADAS11": [12.0, 22.0, 9.0, 8.5],
            "ADAS13": [18.0, 32.0, 14.0, 13.0],
            "FAQ": [2.0, 12.0, 1.0, 0.0],
        })
        sample_df.to_csv(csv_path, index=False)

        records = ADNIClinicalParser.parse_adni_merge(csv_path, max_visits=4)
        assert len(records) == 2

        # Patient 1 converted to Dementia
        p1 = next(r for r in records if r["patient_id"] == "002_S_0295")
        assert p1["is_converter"] == 1
        assert p1["num_observed_visits"] == 2
        assert p1["cognitive"].shape == (4, 5)

        # Patient 2 remained stable MCI
        p2 = next(r for r in records if r["patient_id"] == "002_S_0413")
        assert p2["is_converter"] == 0

        # Dataset item access
        dataset = ADNIMultimodalDataset(csv_path=csv_path, target_mri_shape=(8, 8, 8), max_visits=4)
        assert len(dataset) == 2
        item = dataset[0]
        assert "mri" in item
        assert "cognitive" in item
        assert "ehr" in item
        assert item["mri"].shape == (4, 1, 8, 8, 8)
        assert item["label"].numel() == 1


def test_oasis_clinical_parser_and_dataset():
    """Verify parsing of sample oasis_longitudinal.csv and dataset creation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "oasis_longitudinal.csv")
        sample_df = pd.DataFrame({
            "Subject ID": ["OAS2_0001", "OAS2_0001", "OAS2_0002", "OAS2_0002"],
            "MRI ID": ["OAS2_0001_MR1", "OAS2_0001_MR2", "OAS2_0002_MR1", "OAS2_0002_MR2"],
            "Group": ["Converted", "Converted", "Nondemented", "Nondemented"],
            "Visit": [1, 2, 1, 2],
            "MR Delay": [0, 400, 0, 700],
            "M/F": ["M", "M", "F", "F"],
            "Age": [78, 79, 82, 84],
            "EDUC": [16, 16, 12, 12],
            "SES": [2, 2, 3, 3],
            "MMSE": [28.0, 22.0, 29.0, 29.0],
            "CDR": [0.5, 1.0, 0.0, 0.0],
            "eTIV": [1400.0, 1390.0, 1500.0, 1490.0],
            "nWBV": [0.72, 0.68, 0.77, 0.76],
            "ASF": [1.25, 1.26, 1.17, 1.18],
        })
        sample_df.to_csv(csv_path, index=False)

        records = OASISClinicalParser.parse_oasis_longitudinal(csv_path, max_visits=4)
        assert len(records) == 2

        # Patient 1 converted
        p1 = next(r for r in records if r["patient_id"] == "OAS2_0001")
        assert p1["is_converter"] == 1

        # Patient 2 stable
        p2 = next(r for r in records if r["patient_id"] == "OAS2_0002")
        assert p2["is_converter"] == 0

        # Dataset item access
        dataset = OASISMultimodalDataset(csv_path=csv_path, target_mri_shape=(8, 8, 8), max_visits=4)
        assert len(dataset) == 2
        item = dataset[0]
        assert "mri" in item
        assert item["mri"].shape == (4, 1, 8, 8, 8)


def test_adni_oasis_in_dataset_registry():
    """Verify that ADNI and OASIS are registered in DatasetRegistry."""
    available = DatasetRegistry.list_available()
    assert "adni" in available
    assert "oasis" in available

    mgr_adni = DatasetRegistry.get("adni", mri_shape=(8, 8, 8))
    assert isinstance(mgr_adni, ADNIDatasetManager)
    meta = mgr_adni.get_metadata()
    assert meta["num_classes"] == 2

    mgr_oasis = DatasetRegistry.get("oasis", mri_shape=(8, 8, 8))
    assert isinstance(mgr_oasis, OASISDatasetManager)
    meta_o = mgr_oasis.get_metadata()
    assert meta_o["num_classes"] == 2
