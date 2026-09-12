"""
Algorithm registry service.
"""

from typing import List
from backend.fl_engine.algorithms import ALGORITHM_REGISTRY
from backend.app.schemas.algorithm import AlgorithmInfo


class AlgorithmService:
    DESCRIPTIONS = {
        "fedavg": "Federated Averaging (McMahan et al., 2017) baseline sample-weighted aggregation.",
        "fedprox": "FedProx (Li et al., 2020) with proximal loss penalty for non-IID stabilization.",
        "fedavgm": "FedAvg with Server Momentum (Hsu et al., 2019) for accelerating non-IID convergence.",
        "scaffold": "SCAFFOLD (Karimireddy et al., 2020) with control variates for client drift correction.",
        "fedxneuro": "Fed-XNeuro Explainable Multimodal Federated Learning with Differential Privacy.",
        "fedxneuro_personalized": "Personalized Fed-XNeuro with shared backbone and hospital-specific risk heads.",
    }

    DISPLAY_NAMES = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedavgm": "FedAvgM",
        "scaffold": "SCAFFOLD",
        "fedxneuro": "Fed-XNeuro",
        "fedxneuro_personalized": "Fed-XNeuro Personalized",
    }

    @classmethod
    def list_algorithms(cls) -> List[AlgorithmInfo]:
        unique_keys = sorted(list(set(k for k in ALGORITHM_REGISTRY.keys() if "_" not in k or k in ["fedxneuro_personalized"])))
        res = []
        for k in unique_keys:
            res.append(
                AlgorithmInfo(
                    name=k,
                    display_name=cls.DISPLAY_NAMES.get(k, k.upper()),
                    description=cls.DESCRIPTIONS.get(k, "Federated optimization algorithm"),
                    category="proposed" if "fedxneuro" in k else "baseline",
                )
            )
        return res
