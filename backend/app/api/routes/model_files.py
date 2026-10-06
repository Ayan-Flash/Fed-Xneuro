"""
Model File Discovery API — scans the models/ directory tree for available
checkpoint files (.pt, .h5, .pth, .onnx, .pb, .safetensors) and returns
them as selectable options for the frontend model switcher.
"""

import os
import glob
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/model-files", tags=["model-files"])

# Supported model file extensions
SUPPORTED_EXTENSIONS = {".pt", ".pth", ".h5", ".hdf5", ".onnx", ".pb", ".safetensors"}

# Root directory to scan for model files
WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models")


class ModelFileInfo(BaseModel):
    """Schema representing a discovered model checkpoint file."""
    id: str                          # Unique slug id derived from filename
    filename: str                    # Original filename (e.g. "fedxneuro_best.pt")
    display_name: str                # Human-readable name
    filepath: str                    # Relative path from models/ root
    extension: str                   # File extension (.pt, .h5, etc.)
    size_bytes: int                  # File size in bytes
    size_display: str                # Human-readable size (e.g. "5.7 MB")
    modified_at: str                 # Last modified timestamp
    category: str                    # auto-detected: "global", "checkpoint", "client", "root"


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _slug_from_filename(filename: str) -> str:
    """Generate a URL-safe slug id from a filename."""
    name = os.path.splitext(filename)[0]
    return name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _display_name_from_filename(filename: str) -> str:
    """Generate a human-readable display name from a filename."""
    name = os.path.splitext(filename)[0]
    # Replace underscores/hyphens with spaces, title-case
    return name.replace("_", " ").replace("-", " ").title()


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


def scan_model_files() -> List[ModelFileInfo]:
    """Recursively scans the models/ directory for supported checkpoint files."""
    results: List[ModelFileInfo] = []
    seen_ids: set = set()

    if not os.path.isdir(MODELS_DIR):
        return results

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, MODELS_DIR).replace("\\", "/")

            slug = _slug_from_filename(fname)
            # Ensure unique id by appending category if needed
            category = _detect_category(rel_path)
            unique_id = f"{category}_{slug}" if slug in seen_ids else slug
            seen_ids.add(unique_id)

            try:
                stat = os.stat(full_path)
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y %I:%M %p")
            except OSError:
                size = 0
                mtime = "Unknown"

            results.append(
                ModelFileInfo(
                    id=unique_id,
                    filename=fname,
                    display_name=_display_name_from_filename(fname),
                    filepath=rel_path,
                    extension=ext,
                    size_bytes=size,
                    size_display=_format_size(size),
                    modified_at=mtime,
                    category=category,
                )
            )

    return results


@router.get("", response_model=List[ModelFileInfo])
def list_model_files():
    """
    Returns all available model checkpoint files discovered in the models/ directory.
    Scans recursively for .pt, .pth, .h5, .hdf5, .onnx, .pb, and .safetensors files.
    """
    return scan_model_files()
