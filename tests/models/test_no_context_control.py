"""Unit tests for No-Context Control Baseline & Leakage Auditor (Prompt 36)."""

import pytest
import torch

from src.models.multimodal_baseline import MultimodalBaseline
from src.models.no_context_control import NoContextControlModel
from src.context.leakage_checker import ContextLeakageAuditor


def test_no_context_control_model_forward():
    """Test forward pass of NoContextControlModel matching base model visual output."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=64)
    control_model = NoContextControlModel(base_model=base_model, context_enabled=False)

    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)

    output = control_model(hands=hands, pose=pose)

    assert "logits" in output
    assert output["logits"].dim() == 3
    assert output["logits"].shape[0] == 2
    assert output["logits"].shape[2] == 20


def test_no_context_control_config_flag_enforcement():
    """Assert raising ValueError if context_enabled=True is passed to control model."""
    base_model = MultimodalBaseline(vocab_size=20)
    with pytest.raises(ValueError):
        NoContextControlModel(base_model=base_model, context_enabled=True)


def test_automated_zero_context_leakage_audit():
    """Run ContextLeakageAuditor.assert_zero_context_leakage check on control model."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=64)
    control_model = NoContextControlModel(base_model=base_model, context_enabled=False)

    batch = {
        "hands": torch.randn(2, 16, 126),
        "pose": torch.randn(2, 16, 132)
    }

    report = ContextLeakageAuditor.assert_zero_context_leakage(control_model, batch)

    assert report["bitwise_identical_logits"] is True
    assert report["max_logit_difference"] == 0.0
    assert report["zero_context_gradient"] is True
    assert report["audit_passed"] is True
