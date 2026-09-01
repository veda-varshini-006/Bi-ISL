"""Unit tests for the Bi-ISL Reproducibility Subsystem."""

import json
import os
import random
import tempfile
import pytest

from src.utils.reproducibility import (
    set_seed,
    compute_dict_hash,
    compute_file_hash,
    get_git_info,
    get_device_info,
    get_dependency_snapshot,
    capture_environment_metadata,
    save_experiment_metadata
)


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(1234, deterministic=True)
    val_a = random.randint(0, 1000000)

    set_seed(1234, deterministic=True)
    val_b = random.randint(0, 1000000)

    assert val_a == val_b, "Python random generator did not yield reproducible results."

    try:
        import numpy as np
        set_seed(1234, deterministic=True)
        arr_a = np.random.rand(5)

        set_seed(1234, deterministic=True)
        arr_b = np.random.rand(5)

        assert np.array_equal(arr_a, arr_b), "NumPy random generator did not yield reproducible results."
    except ImportError:
        pass


def test_compute_dict_hash():
    """Test deterministic dictionary hashing."""
    config_a = {"learning_rate": 0.001, "batch_size": 32, "model_name": "BiISL-V1"}
    config_b = {"model_name": "BiISL-V1", "batch_size": 32, "learning_rate": 0.001}

    hash_a = compute_dict_hash(config_a)
    hash_b = compute_dict_hash(config_b)

    assert len(hash_a) == 64, "Hash should be a 64-character SHA-256 string."
    assert hash_a == hash_b, "Dict hashing should be invariant to key insertion order."


def test_compute_file_hash():
    """Test file hashing on a temporary manifest file."""
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
        tmp.write("sample_id,signer_id,label\n1,signer_01,hello\n2,signer_02,thanks\n")
        tmp_path = tmp.name

    try:
        f_hash = compute_file_hash(tmp_path)
        assert len(f_hash) == 64, "File hash should be a 64-character SHA-256 string."
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    assert compute_file_hash("non_existent_file.txt") == "file_not_found"


def test_get_git_info():
    """Test git information retrieval."""
    git_info = get_git_info()
    assert "commit_sha" in git_info
    assert "branch" in git_info
    assert "is_dirty" in git_info
    assert isinstance(git_info["is_dirty"], bool)


def test_get_device_info():
    """Test hardware device information capture."""
    dev_info = get_device_info()
    assert "platform" in dev_info
    assert "python_version" in dev_info
    assert "cuda_available" in dev_info


def test_get_dependency_snapshot():
    """Test dependency snapshot generation."""
    deps = get_dependency_snapshot()
    assert "numpy" in deps
    assert "torch" in deps


def test_capture_environment_metadata():
    """Test capture of full environment metadata dictionary."""
    model_cfg = {"hidden_dim": 512, "num_layers": 6}
    meta = capture_environment_metadata(seed=42, model_config=model_cfg)

    assert meta["seed"] == 42
    assert "timestamp_utc" in meta
    assert "hashes" in meta
    assert meta["hashes"]["model_config_hash"] == compute_dict_hash(model_cfg)


def test_save_experiment_metadata():
    """Test saving metadata to metadata.json."""
    model_cfg = {"hidden_dim": 256}
    meta = capture_environment_metadata(seed=99, model_config=model_cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = save_experiment_metadata(meta, tmp_dir)
        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == "metadata.json"

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["seed"] == 99
        assert loaded_data["model_config"]["hidden_dim"] == 256
