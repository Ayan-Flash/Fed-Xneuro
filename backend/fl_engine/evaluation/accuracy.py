from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def calculate_accuracy(
    model: nn.Module,
    dataset_or_loader: Dataset | DataLoader,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> float:
    """
    Computes top-1 classification accuracy (%) on the given dataset or DataLoader.
    """
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    if isinstance(dataset_or_loader, Dataset):
        if len(dataset_or_loader) == 0:
            return 0.0
        data_loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
    else:
        data_loader = dataset_or_loader

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(dev), target.to(dev)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += data.size(0)

    if total == 0:
        return 0.0

    return (correct / total) * 100.0
