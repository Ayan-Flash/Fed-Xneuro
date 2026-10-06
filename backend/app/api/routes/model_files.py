"""
Model File Discovery API — scans the models/ directory tree for available
checkpoint files (.pt, .h5, .pth, .onnx, .pb, .safetensors) and returns
them as selectable options with rich architectural metadata.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.model_inspector import (
    get_models_dir,
    inspect_model_file,
    find_model_file,
    clear_model_cache,
    SUPPORTED_EXTENSIONS,
)

router = APIRouter(prefix="/model-files", tags=["model-files"])


class ModelFileInfo(BaseModel):
    """Schema representing a discovered model checkpoint file."""
    id: str                          # Unique slug id derived from filename
    filename: str                    # Original filename (e.g. "best_fedxneuro_model.pt")
    display_name: str                # Human-readable name
    filepath: str                    # Relative path from models/ root
    extension: str                   # File extension (.pt, .h5, etc.)
    size_bytes: int                  # File size in bytes
    size_display: str                # Human-readable size (e.g. "9.6 MB")
    modified_at: str                 # Last modified timestamp
    category: str                    # auto-detected: "global", "checkpoint", "client", "root"
    badge: Optional[str] = None      # Dynamic performance/type badge
    architecture: Optional[str] = None # Architectural description
    architecture_type: Optional[str] = None # Category (Multimodal, Vision, etc.)
    param_count: Optional[str] = None # Formatted parameter count if PyTorch
    round: Optional[int] = None      # Training round / epoch if stored in checkpoint
    accuracy: Optional[float] = None # Validation accuracy percentage (0-100)
    macro_f1: Optional[float] = None # Macro F1 score percentage (0-100)
    is_latest: Optional[bool] = False # True for the most recently pasted/trained model file


def _slug_from_filename(filename: str) -> str:
    """Generate a URL-safe slug id from a filename."""
    name = os.path.splitext(filename)[0]
    return name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _detect_category(rel_path: str) -> str:
    """Detect which sub-folder category a model belongs to."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) > 1:
        folder = parts[0].lower()
        if "global" in folder:
            return "global"
        elif "checkpoint" in folder:
            return "checkpoint"
        elif "client" in folder:
            return "client"
        else:
            return folder
    return "root"


def scan_model_files(force_rescan: bool = False) -> List[ModelFileInfo]:
    """Recursively scans the models/ directory for supported checkpoint files."""
    if force_rescan:
        clear_model_cache()

    results: List[ModelFileInfo] = []
    seen_ids: set = set()
    models_dir = get_models_dir()

    if not os.path.isdir(models_dir):
        return results

    raw_items: List[Dict[str, Any]] = []

    for root, _dirs, files in os.walk(models_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, models_dir).replace("\\", "/")

            slug = _slug_from_filename(fname)
            category = _detect_category(rel_path)
            unique_id = f"{category}_{slug}" if slug in seen_ids else slug
            seen_ids.add(unique_id)

            mtime = 0.0
            try:
                stat = os.stat(full_path)
                mtime = stat.st_mtime
            except OSError:
                pass

            # Deep inspect model for rich metadata
            try:
                meta = inspect_model_file(full_path)
                display_name = meta.get("display_name", fname)
                badge = meta.get("badge")
                architecture = meta.get("architecture")
                arch_type = meta.get("type")
                param_count = meta.get("param_count")
                size_bytes = meta.get("size_bytes", 0)
                size_display = meta.get("size_display", "Unknown")
                modified_at = meta.get("modified_at", "Unknown")
                round_val = meta.get("round")
                accuracy_val = meta.get("accuracy")
                macro_f1_val = meta.get("macro_f1")
            except Exception:
                try:
                    stat = os.stat(full_path)
                    size_bytes = stat.st_size
                    modified_at = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y %I:%M %p")
                except OSError:
                    size_bytes = 0
                    modified_at = "Unknown"
                display_name = fname
                badge = f"Checkpoint ({ext.upper()})"
                architecture = f"Model checkpoint ({fname})"
                arch_type = "Neural Network"
                param_count = None
                size_display = f"{size_bytes / (1024*1024):.1f} MB" if size_bytes > 0 else "0 B"
                round_val = None
                accuracy_val = None
                macro_f1_val = None

            raw_items.append({
                "id": unique_id,
                "filename": fname,
                "display_name": display_name,
                "filepath": rel_path,
                "extension": ext,
                "size_bytes": size_bytes,
                "size_display": size_display,
                "modified_at": modified_at,
                "category": category,
                "badge": badge,
                "architecture": architecture,
                "architecture_type": arch_type,
                "param_count": param_count,
                "round": round_val,
                "accuracy": accuracy_val,
                "macro_f1": macro_f1_val,
                "mtime": mtime,
            })

    if not raw_items:
        return []

    # Identify the latest modified file to flag it as is_latest
    max_mtime = max(item["mtime"] for item in raw_items)

    # Sort items: newest modified first, so user immediately sees newly pasted models at top
    raw_items.sort(key=lambda x: x["mtime"], reverse=True)

    for item in raw_items:
        is_latest = (item["mtime"] == max_mtime) and (len(raw_items) > 1)
        results.append(
            ModelFileInfo(
                id=item["id"],
                filename=item["filename"],
                display_name=item["display_name"],
                filepath=item["filepath"],
                extension=item["extension"],
                size_bytes=item["size_bytes"],
                size_display=item["size_display"],
                modified_at=item["modified_at"],
                category=item["category"],
                badge=item["badge"],
                architecture=item["architecture"],
                architecture_type=item["architecture_type"],
                param_count=item["param_count"],
                round=item["round"],
                accuracy=item["accuracy"],
                macro_f1=item["macro_f1"],
                is_latest=is_latest,
            )
        )

    return results


@router.get("", response_model=List[ModelFileInfo])
def list_model_files(rescan: bool = False):
    """
    Returns all available model checkpoint files discovered in the models/ directory.
    Scans recursively for .pt, .pth, .h5, .hdf5, .onnx, .pb, and .safetensors files.
    Pass ?rescan=true to flush inspection cache.
    """
    return scan_model_files(force_rescan=rescan)


@router.post("/rescan", response_model=List[ModelFileInfo])
def rescan_model_files():
    """
    Forces an immediate cache flush and deep rescan of the models/ directory.
    Instantly returns all discovered checkpoint models including newly pasted files.
    """
    return scan_model_files(force_rescan=True)


@router.get("/{filename}/info", response_model=Dict[str, Any])
def get_model_file_info(filename: str):
    """Returns deep inspected architecture and weights metadata for a specific model file."""
    path = find_model_file(filename)
    if not path:
        raise HTTPException(status_code=404, detail=f"Model file '{filename}' not found in models directory")
    return inspect_model_file(path)
