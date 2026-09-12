"""
Dataset registry service.
"""

from typing import List
from backend.fl_engine.datasets import DatasetRegistry
from backend.app.schemas.algorithm import DatasetInfo


class DatasetService:
    DESCRIPTIONS = {
        "mnist": "MNIST 60k grayscale handwritten digits (28x28).",
        "cifar10": "CIFAR-10 60k color natural images (32x32) across 10 classes.",
        "fashion_mnist": "Fashion-MNIST 60k grayscale clothing items (28x28).",
        "multimodal": "ADNI-style longitudinal cohort with 3D MRI, cognitive scores, and EHR timelines.",
        "adni_mci": "ADNI-style MCI progression cohort with realistic clinical follow-ups.",
    }

    DISPLAY_NAMES = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "fashion_mnist": "Fashion-MNIST",
        "multimodal": "Multimodal MCI Cohort (ADNI-style)",
        "adni_mci": "ADNI MCI Progression",
    }

    @classmethod
    def list_datasets(cls) -> List[DatasetInfo]:
        unique_keys = sorted(list(set(DatasetRegistry.list_available())))
        res = []
        for k in unique_keys:
            res.append(
                DatasetInfo(
                    name=k,
                    display_name=cls.DISPLAY_NAMES.get(k, k.upper()),
                    description=cls.DESCRIPTIONS.get(k, "Dataset for federated experiments"),
                    num_classes=2 if "multimodal" in k or "adni" in k else 10,
                    data_type="multimodal" if "multimodal" in k or "adni" in k else "image",
                )
            )
        return res
