"""
TRAIN / Datasets / Preprocessing Module.

Standardized 3D neuroimaging preprocessing pipeline for structural T1 MRI scans:
1. NIfTI (.nii, .nii.gz) loading via nibabel.
2. 3D spatial isotropic resampling to target voxel shapes via scipy.ndimage.
3. Intracranial brain extraction (skull stripping) via intensity thresholding & morphology.
4. Robust voxel intensity normalization (percentile clipping + min-max scaling to [0, 1]).
"""

import os
from typing import Tuple
import numpy as np
import scipy.ndimage

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_nifti_volume(filepath: str) -> np.ndarray:
    """Loads 3D volumetric MRI scan from NIfTI file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MRI file not found: {filepath}")

    if nib is None:
        raise ImportError("nibabel is required. Install via: pip install nibabel")

    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        data = data[..., 0]
    elif data.ndim > 4:
        raise ValueError(f"Unsupported dimensions: {data.ndim} for {filepath}")

    return data


def resample_3d_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (16, 16, 16),
    order: int = 1,
) -> np.ndarray:
    """Resamples 3D volume to target spatial shape (D, H, W)."""
    if volume.shape == target_shape:
        return volume.astype(np.float32)

    zoom_factors = [target_shape[i] / float(volume.shape[i]) for i in range(3)]
    resampled = scipy.ndimage.zoom(volume, zoom=zoom_factors, order=order)
    return resampled.astype(np.float32)


def intensity_normalize_mri(
    volume: np.ndarray,
    method: str = "minmax",
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Normalizes MRI voxel intensities to [0, 1] with percentile outlier clipping."""
    p_low, p_high = np.percentile(volume, clip_percentiles)
    clipped = np.clip(volume, p_low, p_high)

    if method == "minmax":
        c_min, c_max = clipped.min(), clipped.max()
        normalized = (clipped - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(clipped)
    elif method == "zscore":
        mean, std = clipped.mean(), clipped.std()
        normalized = (clipped - mean) / std if std > 0 else clipped - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32)


def extract_brain_mask(volume: np.ndarray, threshold_ratio: float = 0.15) -> np.ndarray:
    """Isolates brain parenchyma from background air and skull."""
    norm_vol = intensity_normalize_mri(volume, method="minmax")
    mask = norm_vol > threshold_ratio
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    mask = scipy.ndimage.binary_opening(mask, structure=struct)
    mask = scipy.ndimage.binary_closing(mask, structure=struct)
    return mask.astype(bool)


def load_and_resample_mri(
    filepath: str,
    target_shape: Tuple[int, int, int] = (16, 16, 16),
    normalize: bool = True,
    apply_mask: bool = False,
) -> np.ndarray:
    """End-to-end preprocessing for a 3D structural MRI volume."""
    raw = load_nifti_volume(filepath)
    if apply_mask:
        raw = raw * extract_brain_mask(raw)
    if normalize:
        raw = intensity_normalize_mri(raw)
    return resample_3d_volume(raw, target_shape=target_shape)
