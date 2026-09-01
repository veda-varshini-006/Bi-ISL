"""Unit tests for UnseenSignerProtocol (Prompt 53)."""

import os
import tempfile
import pytest

from src.data.unseen_signer_protocol import UnseenSignerProtocol


def test_unseen_signer_protocol_anonymization():
    """Test anonymization mapping of held-out signer IDs."""
    signers = ["signer_07", "signer_08", "signer_09"]
    protocol = UnseenSignerProtocol(held_out_signers=signers, anonymize=True)

    assert len(protocol.anonymization_map) == 3
    assert protocol.anonymization_map["signer_07"] == "Anonymous_Signer_A"
    assert protocol.anonymization_map["signer_08"] == "Anonymous_Signer_B"
    assert protocol.anonymization_map["signer_09"] == "Anonymous_Signer_C"


def test_audit_zero_leakage_pass_and_fail():
    """Test strict zero-leakage auditing passes on disjoint sets and fails on overlap."""
    protocol = UnseenSignerProtocol(held_out_signers=["signer_09", "signer_10"])

    train_signers = ["signer_01", "signer_02", "signer_03", "signer_04"]
    val_signers = ["signer_05", "signer_06", "signer_07", "signer_08"]
    test_signers = ["signer_09", "signer_10"]

    # Must pass cleanly
    assert protocol.audit_zero_leakage(train_signers, val_signers, test_signers) is True

    # Must fail on leakage
    leaky_train = ["signer_01", "signer_09"]
    with pytest.raises(ValueError, match="Data leakage detected"):
        protocol.audit_zero_leakage(leaky_train, val_signers, test_signers)


def test_evaluate_unseen_signers():
    """Test evaluation of pre/post adaptation performance across held-out signers."""
    signers = ["signer_08", "signer_09", "signer_10"]
    protocol = UnseenSignerProtocol(held_out_signers=signers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, summary = protocol.evaluate_unseen_signers(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert summary["unseen_signers_count"] == 3
        assert summary["mean_adaptation_gain"] > 0.0


def test_documentation_file_exists():
    """Verify UNSEEN_SIGNER_EVALUATION_PROTOCOL.md exists."""
    doc_path = "./docs/protocols/UNSEEN_SIGNER_EVALUATION_PROTOCOL.md"
    assert os.path.exists(doc_path)
