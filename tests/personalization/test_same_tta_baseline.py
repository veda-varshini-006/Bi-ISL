"""Unit tests for SAMETTABaseline (Prompt 49)."""

import os
import pytest
import torch

from src.personalization.signer_adapter import SignerAdapter
from src.personalization.same_tta_baseline import SAMETTABaseline


def test_same_tta_baseline_loss_computation():
    """Test SAME objective computing entropy loss and feature statistics alignment loss."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    same_base = SAMETTABaseline(adapter=adapter, align_weight=0.1)

    logits = torch.randn(2, 10, 20)
    features = torch.randn(2, 10, 64)

    total_loss, loss_dict = same_base.compute_same_loss(logits, features)

    assert total_loss.item() > 0.0
    assert "entropy_loss" in loss_dict
    assert "align_loss" in loss_dict
    assert "total_same_loss" in loss_dict


def test_same_tta_baseline_adapt_step():
    """Test executing single SAME TTA adaptation step updating adapter parameters."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    same_base = SAMETTABaseline(adapter=adapter, lr=1e-3)

    orig_params = {name: p.clone() for name, p in adapter.named_parameters() if p.requires_grad}
    features = torch.randn(2, 10, 64)
    logits = adapter(features)  # Route through adapter to attach grad_fn

    res = same_base.adapt_same_step(logits, features)

    assert res["status"] == "SAME_TTA_STEP_COMMITTED"
    assert res["step_count"] == 1

    param_changed = any(
        not torch.allclose(p, orig_params[name])
        for name, p in adapter.named_parameters()
        if p.requires_grad
    )
    assert param_changed is True


def test_documentation_file_exists():
    """Verify SAME_TTA_BASELINE_REPRODUCTION.md exists."""
    doc_path = "./docs/baselines/SAME_TTA_BASELINE_REPRODUCTION.md"
    assert os.path.exists(doc_path)
