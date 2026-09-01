"""Unit tests for UGSAGate (Prompt 45)."""

import os
import tempfile
import json
import pytest

from src.personalization.ugsa_gate import UGSAGate


def test_ugsa_gate_accept_decision():
    """Test gate accepts valid high confidence, high consensus, and safe step."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_p = os.path.join(tmp_dir, "decisions.jsonl")
        gate = UGSAGate(tau_p=0.85, tau_q=0.75, epsilon=0.05, log_file_path=log_p)

        res = gate.evaluate(p_t=0.90, q_t=0.80, safety_delta=0.02, signer_id="signer_01")

        assert res["gate_decision"] == 1
        assert "CONFIDENCE_CONSENSUS_AND_SAFETY_VERIFIED" in res["accept_reason"]
        assert res["reject_reason"] == ""


def test_ugsa_gate_reject_reasons():
    """Test gate rejects low confidence, low consensus, or safety degradation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_p = os.path.join(tmp_dir, "decisions.jsonl")
        gate = UGSAGate(tau_p=0.85, tau_q=0.75, epsilon=0.05, log_file_path=log_p)

        res_p = gate.evaluate(p_t=0.50, q_t=0.80, safety_delta=0.02, signer_id="signer_01")
        assert res_p["gate_decision"] == 0
        assert "LOW_CONFIDENCE" in res_p["reject_reason"]

        res_q = gate.evaluate(p_t=0.90, q_t=0.50, safety_delta=0.02, signer_id="signer_01")
        assert res_q["gate_decision"] == 0
        assert "LOW_CONSENSUS" in res_q["reject_reason"]

        res_safe = gate.evaluate(p_t=0.90, q_t=0.80, safety_delta=0.10, signer_id="signer_01")
        assert res_safe["gate_decision"] == 0
        assert "SAFETY_DEGRADATION" in res_safe["reject_reason"]


def test_ugsa_gate_jsonl_logging():
    """Verify decision records are logged to JSONL file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_p = os.path.join(tmp_dir, "decisions.jsonl")
        gate = UGSAGate(log_file_path=log_p)

        gate.evaluate(p_t=0.90, q_t=0.80, safety_delta=0.01, signer_id="signer_99")

        assert os.path.exists(log_p)
        with open(log_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["signer_id"] == "signer_99"
            assert rec["p_t"] == 0.90


def test_documentation_file_exists():
    """Verify UGSA_GATE_DECISION_LOGGING.md exists."""
    doc_path = "./docs/personalization/UGSA_GATE_DECISION_LOGGING.md"
    assert os.path.exists(doc_path)
