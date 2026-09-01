"""Unit tests for TransactionalPersonalizationManager & Rollback (Prompt 47)."""

import os
import tempfile
import json
import pytest
import torch

from src.personalization.signer_adapter import SignerAdapter
from src.personalization.protected_safety_set import ProtectedSafetySet
from src.personalization.bounded_updater import BoundedSignerUpdater
from src.personalization.transactional_personalization import TransactionalPersonalizationManager


def test_snapshot_and_rollback():
    """Test snapshotting state dict and atomic rollback restoring exact original parameters."""
    adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
    safety_set = ProtectedSafetySet()
    updater = BoundedSignerUpdater(adapter=adapter)
    manager = TransactionalPersonalizationManager(adapter=adapter, safety_set=safety_set, updater=updater)

    manager.snapshot_adapter_state()
    orig_w = adapter.down_proj.weight.clone()

    # Perturb weights
    with torch.no_grad():
        adapter.down_proj.weight.add_(1.5)

    assert not torch.allclose(adapter.down_proj.weight, orig_w)

    # Rollback
    manager.rollback()
    assert torch.allclose(adapter.down_proj.weight, orig_w)


def test_execute_transactional_step_commit():
    """Test successful transaction commit when safety score remains high."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_p = os.path.join(tmp_dir, "trans.jsonl")
        adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
        safety_set = ProtectedSafetySet()
        updater = BoundedSignerUpdater(adapter=adapter)
        manager = TransactionalPersonalizationManager(
            adapter=adapter,
            safety_set=safety_set,
            updater=updater,
            history_log_path=log_p
        )

        dummy_loss = torch.tensor(0.1, requires_grad=True)

        res = manager.execute_transactional_step(
            task_loss=dummy_loss,
            pre_safety_score=20.0,
            post_safety_eval_fn=lambda: 19.8,  # 1% drop < 5% threshold
            signer_id="signer_01"
        )

        assert res["status"] == "COMMITTED_SUCCESSFULLY"
        assert res["rollback_performed"] is False
        assert len(manager.adaptation_history) == 1


def test_catastrophic_bad_update_simulated_rollback():
    """Test simulating a catastrophic bad update triggering atomic rollback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_p = os.path.join(tmp_dir, "trans.jsonl")
        adapter = SignerAdapter(in_dim=64, bottleneck_dim=8)
        safety_set = ProtectedSafetySet()
        updater = BoundedSignerUpdater(adapter=adapter)
        manager = TransactionalPersonalizationManager(
            adapter=adapter,
            safety_set=safety_set,
            updater=updater,
            history_log_path=log_p
        )

        orig_w = adapter.down_proj.weight.clone()
        dummy_loss = torch.tensor(10.0, requires_grad=True)

        # Simulate catastrophic post-update evaluation where safety score drops to 10.0 (50% drop)
        res = manager.execute_transactional_step(
            task_loss=dummy_loss,
            pre_safety_score=20.0,
            post_safety_eval_fn=lambda: 10.0,
            signer_id="signer_01"
        )

        assert res["status"] == "ROLLED_BACK_CATASTROPHIC_DEGRADATION"
        assert res["rollback_performed"] is True
        assert res["degradation_pct"] == 50.0

        # Verify weights were atomically restored
        assert torch.allclose(adapter.down_proj.weight, orig_w)


def test_documentation_file_exists():
    """Verify TRANSACTIONAL_PERSONALIZATION_SPEC.md exists."""
    doc_path = "./docs/personalization/TRANSACTIONAL_PERSONALIZATION_SPEC.md"
    assert os.path.exists(doc_path)
