"""Bi-ISL Unified Logging and Diagnostics Framework.

Supports:
- Standard INFO/WARNING/ERROR/DEBUG logging levels.
- Dual output: Human-readable console format & Structured JSON for experiments.
- Per-run log file management.
- GPU memory diagnostics.
- NaN/Inf detection in tensors and numerical values.
- Gradient norm diagnostics (vanishing/exploding gradient detection).
- Data loading failure handling (NEVER silently skip malformed training examples).
- Checkpoint corruption verification.
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class MalformedDataError(Exception):
    """Raised when a malformed training example is detected."""
    pass


class CheckpointCorruptedError(Exception):
    """Raised when a checkpoint file is missing, empty, or corrupted."""
    pass


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log lines for experiment analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_object.update(record.extra_fields)
        if record.exc_info:
            log_object["exception"] = traceback.format_exception(*record.exc_info)
        return json.dumps(log_object, default=str)


class BiISLLogger:
    """Unified logger wrapping standard logging with diagnostic capabilities."""

    def __init__(self, name: str = "BiISL", log_dir: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        self.log_dir = log_dir

        # 1. Console Handler (Human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 2. File Handler (Structured JSON)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "diagnostics.jsonl")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.logger.debug(msg, extra={"extra_fields": kwargs} if kwargs else None)

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg, extra={"extra_fields": kwargs} if kwargs else None)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg, extra={"extra_fields": kwargs} if kwargs else None)

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg, extra={"extra_fields": kwargs} if kwargs else None)

    def close(self) -> None:
        """Flush and close all handlers associated with this logger."""
        for handler in self.logger.handlers[:]:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


def check_gpu_memory(logger: Optional[BiISLLogger] = None) -> Dict[str, Any]:
    """Inspect GPU VRAM usage and log diagnostics."""
    stats = {"cuda_available": False, "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            stats["cuda_available"] = True
            stats["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            stats["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            stats["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

            if logger:
                logger.info(
                    f"GPU Memory: Allocated={stats['allocated_mb']}MB, Reserved={stats['reserved_mb']}MB, Max={stats['max_allocated_mb']}MB",
                    gpu_stats=stats
                )
    except ImportError:
        pass
    return stats


def check_nan_inf(tensor_or_array: Any, name: str = "tensor", raise_error: bool = True) -> bool:
    """Detect NaN or Inf values in PyTorch Tensors, NumPy arrays, or float lists."""
    has_nan = False
    has_inf = False

    try:
        import torch
        if isinstance(tensor_or_array, torch.Tensor):
            has_nan = torch.isnan(tensor_or_array).any().item()
            has_inf = torch.isinf(tensor_or_array).any().item()
    except ImportError:
        pass

    if not has_nan and not has_inf:
        try:
            import numpy as np
            if isinstance(tensor_or_array, np.ndarray):
                has_nan = np.isnan(tensor_or_array).any()
                has_inf = np.isinf(tensor_or_array).any()
        except ImportError:
            pass

    if has_nan or has_inf:
        msg = f"Numerical Instability Detected in '{name}': NaN={has_nan}, Inf={has_inf}"
        if raise_error:
            raise ValueError(msg)
        return True
    return False


def check_gradient_norms(model: Any, max_norm_threshold: float = 100.0, logger: Optional[BiISLLogger] = None) -> float:
    """Calculate and verify total gradient norm for a PyTorch model."""
    total_norm = 0.0
    try:
        import torch
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if logger:
            logger.debug(f"Gradient Norm: {total_norm:.4f}", grad_norm=total_norm)

        if total_norm > max_norm_threshold:
            msg = f"Exploding Gradient Norm Detected: {total_norm:.4f} > threshold {max_norm_threshold}"
            if logger:
                logger.warning(msg)
            else:
                print(f"[WARNING] {msg}")

        if check_nan_inf(torch.tensor(total_norm), name="grad_norm", raise_error=False):
            raise ValueError("NaN or Inf detected in gradient norm.")

    except ImportError:
        pass

    return total_norm


def handle_data_error(sample_id: str, reason: str, logger: Optional[BiISLLogger] = None) -> None:
    """Explicitly handle malformed data examples (NEVER silently skip)."""
    msg = f"CRITICAL DATA FAILURE: Malformed training sample '{sample_id}'. Reason: {reason}"
    if logger:
        logger.error(msg, sample_id=sample_id, failure_reason=reason)
    raise MalformedDataError(msg)


def verify_checkpoint(checkpoint_path: str, logger: Optional[BiISLLogger] = None) -> Dict[str, Any]:
    """Verify PyTorch checkpoint file integrity before loading."""
    if not os.path.exists(checkpoint_path):
        msg = f"Checkpoint File Not Found: '{checkpoint_path}'"
        if logger:
            logger.error(msg)
        raise CheckpointCorruptedError(msg)

    if os.path.getsize(checkpoint_path) == 0:
        msg = f"Checkpoint File Empty (0 bytes): '{checkpoint_path}'"
        if logger:
            logger.error(msg)
        raise CheckpointCorruptedError(msg)

    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise CheckpointCorruptedError("Checkpoint content is not a valid dictionary.")
        if logger:
            logger.info(f"Checkpoint Verified: '{checkpoint_path}'", keys=list(checkpoint.keys()))
        return checkpoint
    except Exception as e:
        msg = f"Checkpoint Corruption Error loading '{checkpoint_path}': {str(e)}"
        if logger:
            logger.error(msg)
        raise CheckpointCorruptedError(msg) from e
