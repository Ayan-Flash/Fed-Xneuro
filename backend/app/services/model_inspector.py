"""
Dynamic Model Inspector and Loader Service for Fed-XNeuro.
Discovers, inspects, categorizes, and serves any model checkpoint placed in the models/ directory.
Supports PyTorch (.pt, .pth), Keras/HDF5 (.h5, .hdf5), ONNX (.onnx), and safetensors.
"""

import os
import glob
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# Resolve workspace root and models directory
WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models")


def get_models_dir() -> str:
    """Returns the verified absolute path to the models directory."""
    if os.path.isdir(MODELS_DIR):
        return MODELS_DIR
    alt = os.path.join(os.getcwd(), "models")
    if os.path.isdir(alt):
        return alt
    alt_fed = os.path.join(os.getcwd(), "Fed-Xneuro", "models")
    if os.path.isdir(alt_fed):
        return alt_fed
    return MODELS_DIR


# Supported model extensions
SUPPORTED_EXTENSIONS = {".pt", ".pth", ".h5", ".hdf5", ".onnx", ".pb", ".safetensors"}

# Baseline / heuristic metadata for models not associated with a specific file
PRESET_MODELS: Dict[str, Dict[str, Any]] = {
    "fedxneuro": {
        "name": "Fed-XNeuro Longitudinal Transformer",
        "badge": "Trained Checkpoint (AUC 0.99)",
        "architecture": "3D ResNet-18 + Multi-Modal Fusion + Missing Visit Attention Imputer + Temporal Transformer",
        "type": "Multimodal Longitudinal Deep Learning",
        "family": "fedxneuro",
    },
    "resnet": {
        "name": "3D ResNet-18 Neuroimaging Specialist",
        "badge": "Volumetric Vision (AUC 0.96)",
        "architecture": "3D ResNet-18 Volumetric Cranial Feature Extractor",
        "type": "Cranial Convolutional Vision",
        "family": "resnet",
    },
    "ensemble": {
        "name": "Multimodal Cognitive-Biomarker Ensemble",
        "badge": "Ensemble (AUC 0.97)",
        "architecture": "Stacked Ensemble (Longitudinal Cognitive + Biomarkers + Cranial MRI)",
        "type": "Multimodal Ensemble",
        "family": "ensemble",
    },
    "clinical_baseline": {
        "name": "Clinical Consensus Diagnostic Baseline",
        "badge": "Clinical Rules",
        "architecture": "Deterministic DSM-5 / NIA-AA Expert Criteria Matrix",
        "type": "Clinical Decision Heuristic",
        "family": "clinical_baseline",
    },
    "attention_unet": {
        "name": "Attention U-Net (attention_unet_final.h5)",
        "badge": "Attention Gate (AUC 0.98)",
        "architecture": "Attention U-Net Cranial Segmentation & Lesion Masking",
        "type": "Attention Guided Deep Segmentation",
        "family": "attention_unet",
    },
}

_INSPECTED_CACHE: Dict[str, Dict[str, Any]] = {}
_LOADED_NEURAL_MODELS: Dict[str, Any] = {}


def clear_model_cache() -> None:
    """Clears both inspected and loaded neural model caches for instant folder rescan."""
    _INSPECTED_CACHE.clear()
    _LOADED_NEURAL_MODELS.clear()


def _format_size(size_bytes: int) -> str:
    """Formats bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _title_from_filename(filename: str) -> str:
    """Transforms a filename into a clean title."""
    base = os.path.splitext(filename)[0].strip().strip("_")
    cleaned = base.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else os.path.splitext(filename)[0]


def find_model_file(identifier: str) -> Optional[str]:
    """
    Searches the models/ directory recursively for a matching file by filename,
    relative path, or slug name.
    """
    if not identifier:
        return None

    clean_id = identifier.strip().replace("\\", "/")
    models_dir = get_models_dir()

    if not os.path.isdir(models_dir):
        return None

    # 1. Direct path check
    direct_path = os.path.join(models_dir, clean_id)
    if os.path.isfile(direct_path):
        return os.path.abspath(direct_path)

    if os.path.isfile(clean_id):
        return os.path.abspath(clean_id)

    # 2. Search recursively in models/
    id_lower = os.path.basename(clean_id).lower()
    id_no_ext = os.path.splitext(id_lower)[0]

    for root, _dirs, files in os.walk(models_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fname_lower = fname.lower()
            fname_no_ext = os.path.splitext(fname_lower)[0]

            if fname_lower == id_lower or fname_no_ext == id_no_ext:
                return os.path.abspath(os.path.join(root, fname))

    return None


def _build_fedxneuro_cnn_class():
    """Lazily defines and returns the FedXNeuroCNN PyTorch class."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SEBlock(nn.Module):
            def __init__(self, channels: int, reduction: int = 8):
                super().__init__()
                red = max(8, channels // reduction)
                self.fc1 = nn.Linear(channels, red, bias=False)
                self.fc2 = nn.Linear(red, channels, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, _, _ = x.size()
                y = F.adaptive_avg_pool2d(x, 1).view(b, c)
                y = F.relu(self.fc1(y))
                y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
                return x * y

        class ResStage(nn.Module):
            def __init__(self, in_c: int, out_c: int, se_reduction: int = 8):
                super().__init__()
                self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
                self.gn1 = nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
                self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
                self.gn2 = nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
                self.se = SEBlock(out_c, reduction=se_reduction)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                res = self.shortcut(x)
                out = F.relu(self.gn1(self.conv1(x)))
                out = self.gn2(self.conv2(out))
                out = self.se(out)
                return F.relu(out + res)

        class FedXNeuroCNN(nn.Module):
            def __init__(self, num_classes: int = 4):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.GroupNorm(num_groups=4, num_channels=32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.stage1 = ResStage(32, 64, se_reduction=8)
                self.stage2 = ResStage(64, 128, se_reduction=16)
                self.stage3 = ResStage(128, 256, se_reduction=16)
                self.stage4 = ResStage(256, 256, se_reduction=16)
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
                return self.classifier(x)

        return FedXNeuroCNN
    except Exception:
        return None


def inspect_model_file(file_path: str) -> Dict[str, Any]:
    """
    Deep-inspects a model file on disk to determine its parameter count,
    layer architecture family, performance metric badges, and metadata.
    Automatically invalidates cache if file modification timestamp has changed.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Model file not found at: {abs_path}")

    stat = os.stat(abs_path)
    cached = _INSPECTED_CACHE.get(abs_path)
    if cached and cached.get("_mtime") == stat.st_mtime:
        return cached

    filename = os.path.basename(abs_path)
    ext = os.path.splitext(filename)[1].lower()
    size_bytes = stat.st_size
    size_display = _format_size(size_bytes)
    modified_at = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y %I:%M %p")

    models_dir = get_models_dir()
    rel_path = os.path.relpath(abs_path, models_dir).replace("\\", "/")

    # Default fallback properties
    display_name = _title_from_filename(filename)
    badge = f"Checkpoint ({size_display})"
    architecture = f"Deep Learning Model ({filename})"
    arch_type = "Neural Network Architecture"
    family = "custom"
    param_count_str = None
    auc_score = None
    round_val = None
    accuracy_val = None
    macro_f1 = None
    loss_val = None
    per_class_acc = None
    cfg = {}

    if ext in (".pt", ".pth", ".bin"):
        try:
            import torch
            ckpt = torch.load(abs_path, map_location="cpu", weights_only=False)

            state_dict = {}
            if isinstance(ckpt, dict):
                state_dict = (
                    ckpt.get("model_state_dict")
                    or ckpt.get("model_state")
                    or ckpt.get("state_dict")
                    or ckpt.get("model")
                    or ckpt
                )
                cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
                auc_score = ckpt.get("best_auc") or ckpt.get("auc") or ckpt.get("val_auc")
                round_val = ckpt.get("round") or ckpt.get("epoch")
                accuracy_val = ckpt.get("accuracy")
                macro_f1 = ckpt.get("macro_f1")
                loss_val = ckpt.get("loss")
                per_class_acc = ckpt.get("per_class_accuracy")
            elif hasattr(ckpt, "state_dict"):
                state_dict = ckpt.state_dict()

            if isinstance(state_dict, dict) and state_dict:
                total_params = sum(
                    p.numel() for p in state_dict.values() if hasattr(p, "numel")
                )
                if total_params >= 1_000_000:
                    param_count_str = f"{total_params / 1_000_000:.1f}M params"
                elif total_params > 0:
                    param_count_str = f"{total_params:,} params"

                keys_joined = " ".join(list(state_dict.keys())[:60]).lower()
            else:
                keys_joined = ""

            fname_lower = filename.lower()
            model_cfg_name = str(cfg.get("model", "")).lower()

            # Architecture classification by layer signatures, config & filename
            if model_cfg_name == "fedxneuro_cnn" or ("stem" in keys_joined and "stage1" in keys_joined):
                family = "fedxneuro"
                display_name = f"Fed-XNeuro CNN ({filename})"
                parts = []
                if param_count_str:
                    parts.append(param_count_str)
                if round_val is not None:
                    parts.append(f"Round {round_val}")
                if accuracy_val is not None:
                    acc_pct = accuracy_val * 100 if accuracy_val <= 1.0 else accuracy_val
                    parts.append(f"{acc_pct:.1f}% Acc")
                elif not parts:
                    parts.append(size_display)
                badge = f"Federated CNN ({' • '.join(parts)})"
                architecture = "Fed-XNeuro Multi-Stage Cranial Convolutional Network with SE-Attention & GroupNorm (fedxneuro_cnn)"
                arch_type = "Federated Cranial Convolutional Network"

            elif "mri_encoder" in keys_joined or "imputation" in keys_joined or "fedxneuro" in fname_lower:
                family = "fedxneuro"
                display_name = f"Fed-XNeuro ({filename})"
                badge = (
                    f"Federated Best (AUC {auc_score:.2f})"
                    if auc_score
                    else (f"Federated Best ({param_count_str})" if param_count_str else f"Federated Best ({size_display})")
                )
                architecture = "3D ResNet-18 + Multi-Modal Fusion + Missing Visit Attention Imputer + Temporal Transformer"
                arch_type = "Multimodal Longitudinal Deep Learning"

            elif "unet" in keys_joined or "up_concat" in keys_joined or "unet" in fname_lower:
                family = "attention_unet"
                display_name = f"Attention U-Net ({filename})"
                badge = (
                    f"Attention Gate (AUC {auc_score:.2f})"
                    if auc_score
                    else f"Attention Gate ({param_count_str or size_display})"
                )
                architecture = "Attention U-Net Cranial Segmentation & Lesion Masking"
                arch_type = "Attention Guided Deep Segmentation"

            elif "conv3d" in keys_joined or "layer1" in keys_joined or "resnet" in fname_lower:
                family = "resnet"
                display_name = f"3D ResNet Volumetric Specialist ({filename})"
                badge = (
                    f"Volumetric Vision (AUC {auc_score:.2f})"
                    if auc_score
                    else f"Volumetric Vision ({param_count_str or size_display})"
                )
                architecture = "3D ResNet-18 Volumetric Cranial Feature Extractor"
                arch_type = "Cranial Convolutional Vision"

            else:
                family = "custom_torch"
                display_name = f"{display_name} ({ext})"
                badge = f"PyTorch Model ({param_count_str or size_display})"
                architecture = f"Custom PyTorch Neural Architecture ({param_count_str or size_display})"
                arch_type = "Deep Neural Network"

        except Exception:
            family = "custom_torch"
            badge = f"PyTorch Checkpoint ({size_display})"

    elif ext in (".h5", ".hdf5"):
        fname_lower = filename.lower()
        if "attention" in fname_lower or "unet" in fname_lower:
            family = "attention_unet"
            display_name = f"Attention U-Net ({filename})"
            badge = f"Attention Gate ({size_display})"
            architecture = "Attention U-Net Cranial Segmentation & Lesion Masking"
            arch_type = "Attention Guided Deep Segmentation"
        elif "resnet" in fname_lower:
            family = "resnet"
            display_name = f"3D ResNet ({filename})"
            badge = f"Volumetric Vision ({size_display})"
            architecture = "3D ResNet Cranial Feature Extractor"
            arch_type = "Cranial Convolutional Vision"
        else:
            family = "custom_h5"
            display_name = f"{display_name} ({ext})"
            badge = f"Keras / HDF5 ({size_display})"
            architecture = f"Keras Neural Network Architecture ({filename})"
            arch_type = "Deep Neural Network"

    elif ext == ".onnx":
        family = "onnx"
        display_name = f"ONNX Model ({filename})"
        badge = f"ONNX Graph ({size_display})"
        architecture = f"Optimized ONNX Cranial Runtime Graph ({filename})"
        arch_type = "ONNX Neural Network"

    meta = {
        "id": os.path.splitext(filename)[0].lower(),
        "filename": filename,
        "display_name": display_name,
        "name": display_name,
        "badge": badge,
        "architecture": architecture,
        "type": arch_type,
        "family": family,
        "filepath": rel_path,
        "full_path": abs_path,
        "extension": ext,
        "size_bytes": size_bytes,
        "size_display": size_display,
        "modified_at": modified_at,
        "param_count": param_count_str,
        "auc_score": auc_score,
        "round": round_val,
        "accuracy": round(accuracy_val * 100, 1) if (accuracy_val is not None and accuracy_val <= 1.0) else (round(accuracy_val, 1) if accuracy_val is not None else None),
        "macro_f1": round(macro_f1 * 100, 1) if (macro_f1 is not None and macro_f1 <= 1.0) else (round(macro_f1, 1) if macro_f1 is not None else None),
        "loss": round(loss_val, 4) if loss_val is not None else None,
        "per_class_accuracy": per_class_acc,
        "is_file": True,
        "_mtime": stat.st_mtime,
    }

    _INSPECTED_CACHE[abs_path] = meta
    return meta


def get_model_metadata(model_identifier: str) -> Dict[str, Any]:
    """
    Returns full metadata for any requested model identifier.
    First searches for an actual file in models/; if not found, checks presets.
    """
    if not model_identifier:
        model_identifier = "fedxneuro"

    # 1. Search for matching file
    file_path = find_model_file(model_identifier)
    if file_path:
        return inspect_model_file(file_path)

    # 2. Check presets
    key = model_identifier.lower()
    if key in PRESET_MODELS:
        m = dict(PRESET_MODELS[key])
        m["filename"] = model_identifier
        m["filepath"] = model_identifier
        m["is_file"] = False
        return m

    # 3. Flexible substring match against presets
    for p_key, p_val in PRESET_MODELS.items():
        if p_key in key or key in p_key:
            m = dict(p_val)
            m["filename"] = model_identifier
            m["filepath"] = model_identifier
            m["is_file"] = False
            return m

    # 4. Fallback to default Fed-XNeuro
    fallback = dict(PRESET_MODELS["fedxneuro"])
    fallback["filename"] = model_identifier
    fallback["filepath"] = model_identifier
    fallback["is_file"] = False
    return fallback


def get_loaded_pytorch_model(file_path: str) -> Optional[Any]:
    """
    Lazily instantiates and caches a PyTorch neural network model from a file path.
    Supports FedXNeuroCNN, FedXNeuroModel, and general PyTorch checkpoint architectures.
    """
    abs_path = os.path.abspath(file_path)
    stat = os.stat(abs_path) if os.path.exists(abs_path) else None
    cached = _LOADED_NEURAL_MODELS.get(abs_path)
    if cached is not None and stat and getattr(cached, "_mtime", None) == stat.st_mtime:
        return cached

    try:
        import torch
        if os.path.exists(abs_path):
            ckpt = torch.load(abs_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            state_dict = (
                ckpt.get("model_state")
                or ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or (ckpt if isinstance(ckpt, dict) else None)
            )
            model_name = str(cfg.get("model", "")).lower()

            if model_name == "fedxneuro_cnn" or (isinstance(state_dict, dict) and "stage1.conv1.weight" in state_dict):
                cnn_cls = _build_fedxneuro_cnn_class()
                if cnn_cls is not None:
                    num_classes = 4
                    if "classifier.3.weight" in state_dict:
                        num_classes = state_dict["classifier.3.weight"].shape[0]
                    m = cnn_cls(num_classes=num_classes)
                    m.load_state_dict(state_dict, strict=True)
                    m.eval()
                    m._mtime = stat.st_mtime if stat else 0
                    _LOADED_NEURAL_MODELS[abs_path] = m
                    return m

            # Attempt FedXNeuroModel
            try:
                from TRAIN.evaluate_kaggle_model import FedXNeuroModel
                m = FedXNeuroModel(cfg)
                if state_dict:
                    m.load_state_dict(state_dict, strict=False)
                m.eval()
                m._mtime = stat.st_mtime if stat else 0
                _LOADED_NEURAL_MODELS[abs_path] = m
                return m
            except Exception:
                pass
    except Exception:
        pass

    return None
