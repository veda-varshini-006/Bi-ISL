"""Unit tests for the Bi-ISL Typed Hierarchical Configuration System."""

import os
import tempfile
import pytest
from pydantic import ValidationError

from src.utils.config import (
    BiISLConfig,
    load_config,
    apply_override,
    DatasetConfig,
    PreprocessingConfig,
    VisualEncoderConfig,
    DecoderConfig,
    ContextConfig,
    UGSAConfig,
    TrainingConfig,
    EvaluationConfig,
    MobileConfig,
    AvatarConfig,
    ExperimentConfig
)


def test_default_config_creation():
    """Test default BiISLConfig initialization."""
    config = BiISLConfig()
    assert config.dataset.dataset_name == "ISLTranslate"
    assert config.training.batch_size == 32
    assert config.decoder.d_model == 512
    assert config.decoder.num_heads == 8
    assert config.context.enable_context is True
    assert config.ugsa.enable_ugsa is True


def test_all_11_groups_present():
    """Test that all 11 required configuration groups exist."""
    config = BiISLConfig()
    groups = [
        "dataset", "preprocessing", "visualencoder", "decoder", "context",
        "ugsa", "training", "evaluation", "mobile", "avatar", "experiment"
    ]
    for g in groups:
        assert hasattr(config, g), f"Missing configuration group: {g}"


def test_yaml_loading_and_saving():
    """Test loading and exporting YAML configuration files."""
    config = BiISLConfig()
    config.training.batch_size = 64
    config.dataset.dataset_name = "iSign"

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml", encoding="utf-8") as tmp:
        tmp.write(config.to_yaml())
        tmp_path = tmp.name

    try:
        loaded_cfg = load_config(tmp_path)
        assert loaded_cfg.training.batch_size == 64
        assert loaded_cfg.dataset.dataset_name == "iSign"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dot_notation_overrides():
    """Test command-line dot-notation overrides."""
    overrides = [
        "training.batch_size=128",
        "dataset.dataset_name=INCLUDE",
        "context.enable_context=false",
        "training.learning_rate=0.0005"
    ]

    config = load_config(overrides=overrides)
    assert config.training.batch_size == 128
    assert config.dataset.dataset_name == "INCLUDE"
    assert config.context.enable_context is False
    assert config.training.learning_rate == 0.0005


def test_invalid_combination_validation():
    """Test validation errors for invalid hyperparameter combinations."""
    # 1. Invalid decoder head dimension (d_model not divisible by num_heads)
    with pytest.raises(ValidationError):
        DecoderConfig(d_model=512, num_heads=7)

    # 2. Negative batch size
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=-16)

    # 3. Invalid reliability threshold (> 1.0)
    with pytest.raises(ValidationError):
        ContextConfig(reliability_threshold=1.5)

    # 4. Invalid protected rollback tolerance (> 0.5)
    with pytest.raises(ValidationError):
        UGSAConfig(protected_rollback_drop_tolerance=0.8)

    # 5. Negative learning rate
    with pytest.raises(ValidationError):
        UGSAConfig(learning_rate=-0.01)


def test_config_to_dict():
    """Test dictionary export format."""
    config = BiISLConfig()
    cfg_dict = config.to_dict()

    assert isinstance(cfg_dict, dict)
    assert "dataset" in cfg_dict
    assert "training" in cfg_dict
    assert cfg_dict["training"]["batch_size"] == 32
