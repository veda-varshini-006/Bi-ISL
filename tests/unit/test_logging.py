"""Unit tests for the Bi-ISL Unified Logging and Diagnostics Framework."""

import json
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.utils.logging import (
    BiISLLogger,
    check_gpu_memory,
    check_nan_inf,
    check_gradient_norms,
    handle_data_error,
    verify_checkpoint,
    MalformedDataError,
    CheckpointCorruptedError
)


def test_logger_levels_and_json_file():
    """Test BiISLLogger creation, dual output, and structured JSON logs."""
    import logging
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = BiISLLogger(name="TestLogger", log_dir=tmp_dir, level=logging.DEBUG)

        logger.debug("Debug message", step=1)
        logger.info("Info message", epoch=2, loss=0.35)
        logger.warning("Warning message", memory_used_mb=450)
        logger.error("Error message", error_code=500)

        jsonl_path = os.path.join(tmp_dir, "diagnostics.jsonl")
        assert os.path.exists(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 4
        log_0 = json.loads(lines[0])
        assert log_0["level"] == "DEBUG"
        assert log_0["step"] == 1

        log_1 = json.loads(lines[1])
        assert log_1["level"] == "INFO"
        assert log_1["loss"] == 0.35

        logger.close()


def test_nan_inf_detection():
    """Test NaN and Inf detection on PyTorch Tensors and NumPy arrays."""
    clean_tensor = torch.tensor([1.0, 2.0, 3.5])
    assert check_nan_inf(clean_tensor, raise_error=False) is False

    nan_tensor = torch.tensor([1.0, float("nan"), 3.5])
    with pytest.raises(ValueError, match="Numerical Instability Detected"):
        check_nan_inf(nan_tensor, name="nan_tensor", raise_error=True)

    inf_arr = np.array([1.0, np.inf, 3.5])
    with pytest.raises(ValueError, match="Numerical Instability Detected"):
        check_nan_inf(inf_arr, name="inf_array", raise_error=True)


def test_gradient_norm_diagnostics():
    """Test gradient norm computation and exploding gradient warnings."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    inputs = torch.randn(4, 10)
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    norm = check_gradient_norms(model, max_norm_threshold=100.0)
    assert norm > 0.0


def test_handle_data_error_no_silent_skip():
    """Test that malformed data errors are NEVER silently skipped."""
    with pytest.raises(MalformedDataError, match="Malformed training sample 'sample_99'"):
        handle_data_error(sample_id="sample_99", reason="Missing target text annotation")


def test_verify_checkpoint_handling():
    """Test checkpoint verification for missing, empty, and valid files."""
    # 1. Missing file
    with pytest.raises(CheckpointCorruptedError, match="Checkpoint File Not Found"):
        verify_checkpoint("non_existent_ckpt.pt")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 2. Empty file (0 bytes)
        empty_ckpt = os.path.join(tmp_dir, "empty.pt")
        open(empty_ckpt, "w").close()

        with pytest.raises(CheckpointCorruptedError, match="Checkpoint File Empty"):
            verify_checkpoint(empty_ckpt)

        # 3. Valid checkpoint file
        valid_ckpt = os.path.join(tmp_dir, "valid.pt")
        torch.save({"epoch": 10, "state_dict": {"w": torch.tensor([1.0])}}, valid_ckpt)

        loaded_data = verify_checkpoint(valid_ckpt)
        assert loaded_data["epoch"] == 10
