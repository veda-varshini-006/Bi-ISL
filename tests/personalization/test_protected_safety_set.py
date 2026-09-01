"""Unit tests for ProtectedSafetySet (Prompt 44)."""

import os
import pytest

from src.personalization.protected_safety_set import ProtectedSafetySet


def test_protected_safety_set_init_and_version():
    """Test safety set loading, size, version_id, and SHA256 hash."""
    safety_set = ProtectedSafetySet(version_id="v1.0.0")

    assert len(safety_set.samples) == 50
    assert safety_set.version_id == "v1.0.0"
    assert len(safety_set.manifest_hash) == 64  # SHA256 length


def test_audit_non_contamination_pass_and_fail():
    """Test non-contamination auditor passes for disjoint IDs and fails on overlap."""
    safety_set = ProtectedSafetySet()

    clean_test_ids = ["test_sample_001", "test_sample_002", "test_sample_003"]
    assert safety_set.audit_non_contamination(clean_test_ids) is True

    contaminated_test_ids = ["test_sample_001", "safety_ref_005"]
    with pytest.raises(ValueError, match="Contamination Detected"):
        safety_set.audit_non_contamination(contaminated_test_ids)


def test_measure_adaptation_degradation_and_rollback():
    """Test pre/post degradation calculation and emergency rollback trigger."""
    safety_set = ProtectedSafetySet()

    # Minimal degradation (2% drop) -> No rollback
    res1 = safety_set.measure_adaptation_degradation(pre_adaptation_bleu=20.0, post_adaptation_bleu=19.6)
    assert res1["degradation_pct"] == 2.0
    assert res1["rollback_required"] is False

    # Severe degradation (10% drop) -> Emergency rollback
    res2 = safety_set.measure_adaptation_degradation(pre_adaptation_bleu=20.0, post_adaptation_bleu=18.0)
    assert res2["degradation_pct"] == 10.0
    assert res2["rollback_required"] is True


def test_documentation_file_exists():
    """Verify PROTECTED_SAFETY_SET_CONSTRUCTION.md exists."""
    doc_path = "./docs/personalization/PROTECTED_SAFETY_SET_CONSTRUCTION.md"
    assert os.path.exists(doc_path)
