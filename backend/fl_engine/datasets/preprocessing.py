"""
3D Neuroimaging Preprocessing Module for Fed-XNeuro.

Provides standardized volumetric operations for structural T1-weighted brain MRI:
1. NIfTI (.nii, .nii.gz) volume loading via nibabel.
2. 3D spatial isotropic resampling and interpolation via scipy.ndimage.
3. Skull-stripping / intracranial brain extraction mask.
4. Intensity z-score normalization and [0, 1] clipping.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import scipy.ndimage
import torch

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_nifti_volume(filepath: str) -> np.ndarray:
    """
    Loads raw 3D volumetric MRI scan from NIfTI (.nii or .nii.gz) file.
    Returns:
        3D float32 numpy array with voxel intensity values.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MRI file not found: {filepath}")

    if nib is None:
        raise ImportError("nibabel is required to read NIfTI files. Install via pip install nibabel.")

    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)

    # If 4D volume (e.g., multi-echo or time-series), extract first 3D volume
    if data.ndim == 4:
        data = data[..., 0]
    elif data.ndim > 4:
        raise ValueError(f"Unsupported MRI volume dimensions: {data.ndim} for {filepath}")

    return data


def resample_3d_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (16, 16, 16),
    order: int = 1,
) -> np.ndarray:
    """
    Resamples 3D volume to target spatial shape (D, H, W) via spline/trilinear interpolation.
    """
    if volume.shape == target_shape:
        return volume.astype(np.float32)

    zoom_factors = [
        target_shape[i] / float(volume.shape[i]) for i in range(3)
    ]
    resampled = scipy.ndimage.zoom(volume, zoom=zoom_factors, order=order)
    return resampled.astype(np.float32)


def intensity_normalize_mri(
    volume: np.ndarray,
    method: str = "minmax",
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """
    Normalizes MRI voxel intensities to [0, 1] with outlier percentile clipping.
    """
    # Clip extreme outlier voxels (e.g. scanner artifacts)
    p_low, p_high = np.percentile(volume, clip_percentiles)
    clipped = np.clip(volume, p_low, p_high)

    if method == "minmax":
        c_min, c_max = clipped.min(), clipped.max()
        if c_max > c_min:
            normalized = (clipped - c_min) / (c_max - c_min)
        else:
            normalized = np.zeros_like(clipped)
    elif method == "zscore":
        mean, std = clipped.mean(), clipped.std()
        if std > 0:
            normalized = (clipped - mean) / std
        else:
            normalized = clipped - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32)


def extract_brain_mask(
    volume: np.ndarray,
    threshold_ratio: float = 0.15,
) -> np.ndarray:
    """
    Computes a binary brain extraction mask isolating parenchymal tissue
    from background air and cranial bone.
    """
    norm_vol = intensity_normalize_mri(volume, method="minmax")
    mask = norm_vol > threshold_ratio

    # Morphological opening/closing to remove small noise specks
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
    """
    End-to-end preprocessing pipeline for a single 3D MRI volume.
    Returns:
        3D float32 numpy array with shape (target_shape).
    """
    raw_volume = load_nifti_volume(filepath)

    if apply_mask:
        mask = extract_brain_mask(raw_volume)
        raw_volume = raw_volume * mask

    if normalize:
        raw_volume = intensity_normalize_mri(raw_volume)

    resampled = resample_3d_volume(raw_volume, target_shape=target_shape)
    return resampled
