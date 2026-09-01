"""Unit tests for Abstention Mechanism & Clarification Engine (Prompts 55 & 56)."""

import os
import tempfile
import pytest

from src.eval.abstention_mechanism import AbstentionMechanism
from src.eval.clarification_engine import ClarificationEngine


def test_abstention_mechanism_eval():
    """Test abstention triggering on low confidence and risk-coverage curve generation."""
    abstain = AbstentionMechanism(tau_p=0.85, tau_entropy=1.5)

    res_clean = abstain.evaluate_abstention(p_t=0.92, entropy=0.8)
    assert res_clean["should_abstain"] is False
    assert res_clean["status"] == "TRANSLATE"

    res_low_p = abstain.evaluate_abstention(p_t=0.60, entropy=0.8)
    assert res_low_p["should_abstain"] is True
    assert "LOW_VISUAL_CONFIDENCE" in res_low_p["abstain_reasons"]

    curve = abstain.generate_risk_coverage_curve()
    assert len(curve["coverages"]) == 10
    assert len(curve["risks"]) == 10


def test_clarification_engine_repeat_sign():
    """Test generating 'Please repeat the sign' prompt on low visual confidence."""
    engine = ClarificationEngine()
    abstention_info = {"abstain_reasons": ["LOW_VISUAL_CONFIDENCE"]}

    out = engine.generate_clarification_prompt(abstention_info=abstention_info)
    assert out["clarification_message"] == "Please repeat the sign."
    assert out["clarification_type"] == "REPEAT_SIGN"


def test_clarification_engine_disambiguation():
    """Test generating 'Did you mean X or Y?' prompt for top beam candidates."""
    engine = ClarificationEngine()
    abstention_info = {"abstain_reasons": ["HIGH_SEQUENCE_ENTROPY"]}

    out = engine.generate_clarification_prompt(
        abstention_info=abstention_info,
        top_candidates=["Fever", "Cough"],
        domain="GENERAL"
    )
    assert "Did you mean 'Fever' or 'Cough'?" in out["clarification_message"]
    assert out["clarification_type"] == "DISAMBIGUATION"


def test_clarification_engine_safety_guardrails():
    """Test prohibiting unsupported medical/legal advice during clarification."""
    engine = ClarificationEngine()
    abstention_info = {"abstain_reasons": ["HIGH_SEQUENCE_ENTROPY"]}

    out = engine.generate_clarification_prompt(
        abstention_info=abstention_info,
        top_candidates=["Diagnose Patient", "Cough"],
        domain="MEDICAL"
    )
    # 'Diagnose' term must be sanitized to [UNVERIFIED_TERM]
    assert "[UNVERIFIED_TERM]" in out["clarification_message"]
    assert "diagnose patient" not in out["clarification_message"].lower()


def test_usability_metric_logging():
    """Test tracking and logging clarification frequency metric."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ClarificationEngine(log_dir=tmp_dir)
        abstention_info = {"abstain_reasons": ["LOW_VISUAL_CONFIDENCE"]}

        out1 = engine.generate_clarification_prompt(abstention_info=abstention_info)
        out2 = engine.generate_clarification_prompt(abstention_info=abstention_info)

        assert out2["clarification_frequency_pct"] == 100.0
        assert out2["total_queries"] == 2
        assert os.path.exists(os.path.join(tmp_dir, "clarification_usability.jsonl"))


def test_documentation_file_exists():
    """Verify CLARIFICATION_WORKFLOW_SPEC.md exists."""
    doc_path = "./docs/usability/CLARIFICATION_WORKFLOW_SPEC.md"
    assert os.path.exists(doc_path)
