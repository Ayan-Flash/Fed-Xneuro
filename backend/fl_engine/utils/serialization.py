import os
from typing import Dict, Optional
import torch


def save_state_dict(state_dict: Dict[str, torch.Tensor], file_path: str) -> str:
    """
    Saves a model state_dict safely to disk.
    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Detach and move to CPU before saving
    cpu_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    torch.save(cpu_state, file_path)
    return file_path


def load_state_dict(file_path: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Loads a model state_dict safely from disk.
    Uses weights_only=True for security.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {file_path}")
    
    target_device = device or torch.device("cpu")
    state_dict = torch.load(file_path, map_location=target_device, weights_only=True)
    return state_dict
