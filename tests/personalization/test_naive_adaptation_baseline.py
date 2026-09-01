"""Unit tests for NaiveSignerFineTuningBaseline (Prompt 48)."""

import os
import pytest
import torch

from src.personalization.signer_adapter import SignerAdapter
from src.personalization.naive_adaptation_baseline import NaiveSignerFineTuningBaseline


def test_naive_signer_fine_tuning_uncontrolled_step():
    """Test uncontrolled step updating parameters without gate or rollback."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    naive_base = NaiveSignerFineTuningBaseline(adapter=adapter, lr=1e-3)

    orig_params = {name: p.clone() for name, p in adapter.named_parameters() if p.requires_grad}
    dummy_input = torch.randn(2, 64)
    dummy_target = torch.randn(2, 64)

    loss = torch.mean((adapter(dummy_input) - dummy_target) ** 2)
    res = naive_base.adapt_uncontrolled_step(loss)

    assert res["status"] == "UNCONTROLLED_COMMIT"
    assert res["confidence_gate_applied"] is False
    assert res["safety_rollback_applied"] is False
    assert res["step_count"] == 1

    # Weights updated permanently (at least one parameter changed)
    param_changed = any(
        not torch.allclose(p, orig_params[name])
        for name, p in adapter.named_parameters()
        if p.requires_grad
    )
    assert param_changed is True


def test_documentation_file_exists():
    """Verify NAIVE_ADAPTATION_BASELINE_SPEC.md exists."""
    doc_path = "./docs/baselines/NAIVE_ADAPTATION_BASELINE_SPEC.md"
    assert os.path.exists(doc_path)
