from typing import Dict, Any
import torch
import torch.nn as nn


class CommunicationTracker:
    """
    Tracks and estimates communication costs (upload, download, total)
    based on PyTorch model parameters.
    """

    @staticmethod
    def calculate_model_size_bytes(model_or_state_dict: nn.Module | Dict[str, torch.Tensor]) -> int:
        """
        Calculates total model size in bytes based on tensor parameter storage.
        """
        if isinstance(model_or_state_dict, nn.Module):
            state = model_or_state_dict.state_dict()
        else:
            state = model_or_state_dict

        total_bytes = sum(
            tensor.nelement() * tensor.element_size()
            for tensor in state.values()
        )
        return total_bytes

    def __init__(self, model_size_bytes: int = 0) -> None:
        self.model_size_bytes = model_size_bytes
        self.cumulative_upload_bytes = 0
        self.cumulative_download_bytes = 0
        self.cumulative_total_bytes = 0

    def track_round(self, num_selected_clients: int, model_size_bytes: int = 0) -> Dict[str, Any]:
        """
        Records communication for a federated round.
        Download: server distributes global model to selected clients.
        Upload: clients transmit updated local models to server.
        """
        size = model_size_bytes or self.model_size_bytes
        round_download = size * num_selected_clients
        round_upload = size * num_selected_clients
        round_total = round_download + round_upload

        self.cumulative_download_bytes += round_download
        self.cumulative_upload_bytes += round_upload
        self.cumulative_total_bytes += round_total

        return {
            "model_size_bytes": size,
            "model_size_mb": round(size / (1024 * 1024), 4),
            "round_download_bytes": round_download,
            "round_upload_bytes": round_upload,
            "round_total_bytes": round_total,
            "cumulative_download_bytes": self.cumulative_download_bytes,
            "cumulative_upload_bytes": self.cumulative_upload_bytes,
            "cumulative_total_bytes": self.cumulative_total_bytes,
            "cumulative_total_mb": round(self.cumulative_total_bytes / (1024 * 1024), 4),
        }
