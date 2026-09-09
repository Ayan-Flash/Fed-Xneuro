from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def calculate_loss(
    model: nn.Module,
    dataset_or_loader: Dataset | DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> float:
    """
    Computes average loss on the given dataset or DataLoader.
    """
    dev = device or torch.device("cpu")
    crit = criterion or nn.CrossEntropyLoss()
    model.to(dev)
    model.eval()

    if isinstance(dataset_or_loader, Dataset):
        if len(dataset_or_loader) == 0:
            return 0.0
        data_loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
    else:
        data_loader = dataset_or_loader

    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(dev), target.to(dev)
            output = model(data)
            loss = crit(output, target)
            total_loss += loss.item() * data.size(0)
            total += data.size(0)

    if total == 0:
        return 0.0

    return float(total_loss / total)
