from typing import Dict, Any
import torch
from backend.fl_engine.utils.seed import set_seed


def get_system_reproducibility_info() -> Dict[str, Any]:
    """
    Returns diagnostic information about hardware and libraries
    relevant to reproducibility.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    return {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cudnn_enabled": torch.backends.cudnn.enabled if cuda_available else False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic if cuda_available else False,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if cuda_available else False,
    }


__all__ = ["set_seed", "get_system_reproducibility_info"]
