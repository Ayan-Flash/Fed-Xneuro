#!/usr/bin/env python
"""
Fed-XNeuro: Real Dataset Ingestion & Preprocessing CLI.

Validates, preprocesses, and splits real ADNI or OASIS cohorts:
1. Ingests ADNIMERGE.csv or oasis_longitudinal.csv.
2. Ingests and resamples 3D T1-weighted NIfTI MRI scans (.nii, .nii.gz).
3. Enforces strict patient-level splitting (zero longitudinal cross-visit leakage).
4. Caches preprocessed tensors into data/processed/<source>/ for high-performance training.
"""

import os
import sys
import argparse
import json
import glob
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is on PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.fl_engine.datasets.adni_loader import ADNIClinicalParser, ADNIMultimodalDataset
from backend.fl_engine.datasets.oasis_loader import OASISClinicalParser, OASISMultimodalDataset
from backend.fl_engine.datasets.preprocessing import load_and_resample_mri


def ingest_dataset(
    source: str,
    raw_dir: str,
    csv_filename: str,
    mri_subfolder: str,
    output_dir: str,
    mri_shape: tuple = (16, 16, 16),
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 75)
    print(f"  FED-XNEURO: REAL DATASET INGESTION & PREPROCESSING ({source.upper()})")
    print("=" * 75)

    csv_path = os.path.join(raw_dir, csv_filename)
    mri_dir = os.path.join(raw_dir, mri_subfolder)

    print(f"[*] Source Database: {source.upper()}")
    print(f"[*] Raw Directory:   {raw_dir}")
    print(f"[*] Clinical CSV:    {csv_path} ({'FOUND' if os.path.exists(csv_path) else 'NOT FOUND - using fallback'})")
    print(f"[*] MRI Directory:   {mri_dir} ({'FOUND' if os.path.exists(mri_dir) else 'NOT FOUND - using fallback'})")
    print(f"[*] Target 3D Shape: {mri_shape}")
    print(f"[*] Output Cache:    {output_dir}")

    # 1. Instantiate Multimodal Dataset
    if source.lower() == "adni":
        dataset = ADNIMultimodalDataset(
            csv_path=csv_path if os.path.exists(csv_path) else None,
            mri_dir=mri_dir if os.path.exists(mri_dir) else None,
            target_mri_shape=mri_shape,
            seed=seed,
        )
    elif source.lower() == "oasis":
        dataset = OASISMultimodalDataset(
            csv_path=csv_path if os.path.exists(csv_path) else None,
            mri_dir=mri_dir if os.path.exists(mri_dir) else None,
            target_mri_shape=mri_shape,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported source: {source}. Choose 'adni' or 'oasis'.")

    n_total = len(dataset)
    labels = dataset.get_labels()
    converters = int(np.sum(labels == 1))
    stables = int(np.sum(labels == 0))

    print(f"\n[1/4] Cohort Summary:")
    print(f"      Total Patients:        {n_total}")
    print(f"      MCI -> AD Progressors: {converters} ({converters / max(1, n_total) * 100:.1f}%)")
    print(f"      Stable MCI Controls:   {stables} ({stables / max(1, n_total) * 100:.1f}%)")

    # 2. Patient-Level Train/Test Split
    print("\n[2/4] Generating Patient-Level Train/Test Split (80/20)...")
    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(n_total).tolist()
    n_train = int(n_total * 0.80)

    train_indices = shuffled_idx[:n_train]
    test_indices = shuffled_idx[n_train:]

    train_pids = [dataset[i]["patient_id"] for i in train_indices]
    test_pids = [dataset[i]["patient_id"] for i in test_indices]

    # Verify zero leakage
    overlap = set(train_pids).intersection(set(test_pids))
    assert len(overlap) == 0, f"Error: Patient leakage detected across split! Overlap: {overlap}"
    print("      Verification: ZERO cross-visit patient data leakage between train and test sets.")
    print(f"      Train Cohort: {len(train_pids)} patients")
    print(f"      Test Cohort:  {len(test_pids)} patients")

    # 3. Export Cached Tensors
    print("\n[3/4] Packaging & Caching Processed Tensors to Disk...")
    os.makedirs(output_dir, exist_ok=True)

    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    train_save_path = os.path.join(output_dir, "train_cohort.pt")
    test_save_path = os.path.join(output_dir, "test_cohort.pt")
    manifest_path = os.path.join(output_dir, "manifest.json")

    torch.save(train_data, train_save_path)
    torch.save(test_data, test_save_path)

    manifest = {
        "source": source,
        "total_patients": n_total,
        "converters": converters,
        "stables": stables,
        "train_patients": len(train_pids),
        "test_patients": len(test_pids),
        "mri_shape": list(mri_shape),
        "train_cohort_file": train_save_path,
        "test_cohort_file": test_save_path,
        "zero_leakage_verified": True,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"      Saved Train Cohort: {train_save_path}")
    print(f"      Saved Test Cohort:  {test_save_path}")
    print(f"      Saved Manifest:     {manifest_path}")

    print("\n[4/4] Ingestion Completed Successfully!")
    print("=" * 75)
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fed-XNeuro: Real Dataset Ingestion & Preprocessing Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, default="adni", choices=["adni", "oasis"], help="Dataset source")
    parser.add_argument("--raw-dir", type=str, default=None, help="Root folder containing raw CSV and MRI subfolder")
    parser.add_argument("--csv-name", type=str, default=None, help="Filename of the clinical CSV")
    parser.add_argument("--mri-subfolder", type=str, default="mri", help="Subfolder containing 3D NIfTI scans")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for processed tensors")
    parser.add_argument("--mri-shape", nargs=3, type=int, default=[16, 16, 16], help="Target 3D MRI voxel shape (D H W)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient-level split")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    src = args.source.lower()

    if args.raw_dir is None:
        raw_dir = os.path.join(project_root, "data", "raw", src)
    else:
        raw_dir = args.raw_dir

    if args.csv_name is None:
        csv_name = "ADNIMERGE.csv" if src == "adni" else "oasis_longitudinal.csv"
    else:
        csv_name = args.csv_name

    if args.output_dir is None:
        output_dir = os.path.join(project_root, "data", "processed", src)
    else:
        output_dir = args.output_dir

    ingest_dataset(
        source=src,
        raw_dir=raw_dir,
        csv_filename=csv_name,
        mri_subfolder=args.mri_subfolder,
        output_dir=output_dir,
        mri_shape=tuple(args.mri_shape),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
