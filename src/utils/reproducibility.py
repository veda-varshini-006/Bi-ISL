"""Bi-ISL Reproducibility Subsystem.

Provides functionality for seed setting, deterministic hardware configuration,
environment metadata capture, dependency snapshots, git commit tracking,
configuration hashing, and metadata.json generation.
"""

import hashlib
import json
import os
import platform
import random
import sys
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, PyTorch, and configure CUDA determinism."""
    # 1. Python built-in random & hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy seed
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # 3. PyTorch CPU & CUDA seeds
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
    except ImportError:
        pass


def compute_dict_hash(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary (model/experiment config)."""
    if not data:
        return "empty"
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file (dataset manifest / split file)."""
    if not os.path.exists(file_path):
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieve Git commit SHA, branch, and dirty status."""
    git_info = {
        "commit_sha": "unknown",
        "branch": "unknown",
        "is_dirty": False
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        git_info["commit_sha"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        git_info["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        git_info["is_dirty"] = len(status) > 0
    except Exception:
        pass

    return git_info


def get_device_info() -> Dict[str, Any]:
    """Capture hardware device information (CPU, GPU, PyTorch CUDA status)."""
    device_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }

    try:
        import torch
        device_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device_info["device_count"] = torch.cuda.device_count()
            device_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            device_info["pytorch_cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return device_info


def get_dependency_snapshot() -> Dict[str, str]:
    """Capture versions of key installed dependencies."""
    packages = [
        "torch", "torchvision", "numpy", "scipy", "pyyaml", "pydantic",
        "cv2", "mediapipe", "sacrebleu", "transformers", "onnx", "onnxruntime"
    ]
    snapshot = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            snapshot[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            snapshot[pkg] = "not_installed"
    return snapshot


def capture_environment_metadata(
    seed: int = 42,
    model_config: Optional[Dict[str, Any]] = None,
    dataset_manifest_path: Optional[str] = None
) -> Dict[str, Any]:
    """Capture complete execution metadata for an experiment run."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git": get_git_info(),
        "device": get_device_info(),
        "dependencies": get_dependency_snapshot(),
        "hashes": {
            "model_config_hash": compute_dict_hash(model_config or {}),
            "dataset_manifest_hash": compute_file_hash(dataset_manifest_path) if dataset_manifest_path else "not_provided"
        },
        "model_config": model_config or {}
    }
    return metadata


def save_experiment_metadata(metadata: Dict[str, Any], output_dir: str) -> str:
    """Save metadata dictionary to metadata.json in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata_path
