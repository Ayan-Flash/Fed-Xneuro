"""
TRAIN Datasets Package.
Provides clinical parsers, multimodal dataset wrappers, and NIfTI MRI preprocessing
for ADNI and OASIS cohorts.
"""

from TRAIN.datasets.preprocessing import load_and_resample_mri
from TRAIN.datasets.adni_loader import ADNIClinicalParser, ADNIMultimodalDataset, ADNIDatasetManager
from TRAIN.datasets.oasis_loader import OASISClinicalParser, OASISMultimodalDataset, OASISDatasetManager

__all__ = [
    "load_and_resample_mri",
    "ADNIClinicalParser",
    "ADNIMultimodalDataset",
    "ADNIDatasetManager",
    "OASISClinicalParser",
    "OASISMultimodalDataset",
    "OASISDatasetManager",
]
