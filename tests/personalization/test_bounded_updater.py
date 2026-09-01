"""Unit tests for BoundedSignerUpdater (Prompt 46)."""

import os
import pytest
import torch

from src.personalization.signer_adapter import SignerAdapter, AdapterPlacement
from src.personalization.bounded_updater import BoundedSignerUpdater


def test_bounded_updater_step_limit():
    """Test updater enforces gradient step limit and stops updating."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    updater = BoundedSignerUpdater(adapter=adapter, max_steps=2, lr=1e-4)

    dummy_input = torch.randn(2, 64)
    dummy_target = torch.randn(2, 64)

    # Step 1
    loss1 = torch.mean((adapter(dummy_input) - dummy_target) ** 2)
    res1 = updater.update_step(loss1)
    assert res1["updated"] is True
    assert res1["step_count"] == 1

    # Step 2
    loss2 = torch.mean((adapter(dummy_input) - dummy_target) ** 2)
    res2 = updater.update_step(loss2)
    assert res2["updated"] is True
    assert res2["step_count"] == 2

    # Step 3 -> Rejected due to max_steps=2
    loss3 = torch.mean((adapter(dummy_input) - dummy_target) ** 2)
    res3 = updater.update_step(loss3)
    assert res3["updated"] is False
    assert "MAX_STEPS_REACHED" in res3["reason"]


def test_regularized_loss_and_parameter_distance():
    """Test L2 distance measurement and regularization loss computation."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    updater = BoundedSignerUpdater(adapter=adapter, l2_lambda=0.05)

    dist_init = updater.compute_parameter_distance()
    assert dist_init == 0.0

    # Perturb adapter weights slightly
    with torch.no_grad():
        adapter.down_proj.weight.add_(0.1)

    dist_after = updater.compute_parameter_distance()
    assert dist_after > 0.0

    reg_loss = updater.compute_regularization_loss()
    assert reg_loss.item() > 0.0


def test_distance_projection_sphere_bound():
    """Test hard parameter projection back onto distance sphere when R_max is exceeded."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    updater = BoundedSignerUpdater(adapter=adapter, max_dist_r=0.20)

    # Exceed max_dist_r manually
    with torch.no_grad():
        adapter.down_proj.weight.add_(2.0)

    dist_huge = updater.compute_parameter_distance()
    assert dist_huge > 0.20

    dummy_input = torch.randn(2, 64)
    loss = torch.mean(adapter(dummy_input) ** 2)

    res = updater.update_step(loss)
    # Distance should be projected back to max_dist_r = 0.20
    assert res["param_distance"] <= 0.2001


def test_documentation_file_exists():
    """Verify BOUNDED_ADAPTATION_SPEC.md exists."""
    doc_path = "./docs/personalization/BOUNDED_ADAPTATION_SPEC.md"
    assert os.path.exists(doc_path)
