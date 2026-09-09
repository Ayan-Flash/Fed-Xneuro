import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = False) -> None:
    """
    Sets random seeds across Python, NumPy, and PyTorch for reproducible simulations.
    
    Args:
        seed: Integer seed value.
        deterministic_cuda: If True, configures PyTorch backends to enforce
                            deterministic convolution algorithms.
                            Note: Enforcing strict determinism may reduce GPU throughput.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
