"""Unit tests for DecisionTracer (Prompt 58)."""

import os
import tempfile
import pytest

from src.explainability.decision_tracer import DecisionTracer


def test_trace_decision():
    """Test tracing per-example decision metadata and writing JSONL log."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracer = DecisionTracer(log_dir=tmp_dir)

        rec = tracer.trace_decision(
            sample_id="sample_001",
            visual_confidence=0.9123,
            context_gate_alpha=0.4567,
            ugsa_state="STABLE",
            adaptation_accepted=True
        )

        assert rec["sample_id"] == "sample_001"
        assert rec["visual_confidence"] == 0.9123
        assert rec["context_gate_alpha"] == 0.4567
        assert os.path.exists(os.path.join(tmp_dir, "decision_telemetry.jsonl"))
