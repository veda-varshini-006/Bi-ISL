"""Unit tests for E2/E3 Evaluator & Phase 4 Final Report (Prompt 40)."""

import os
import tempfile
import pytest

from src.experiments.e2_e3_evaluator import run_e2_e3_comprehensive_evaluation


def test_e2_e3_comprehensive_evaluation_h1_and_h2():
    """Test running E2/E3 evaluation verifying 5 systems, 5 corruptions, and hypotheses H1/H2."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, report = run_e2_e3_comprehensive_evaluation(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert len(report["systems_evaluated"]) == 5
        assert len(report["corruption_levels"]) == 5

        h1 = report["hypothesis_testing"]["H1_context_gating_efficacy"]
        h2 = report["hypothesis_testing"]["H2_contradiction_robustness"]

        assert h1["h1_passed"] is True
        assert h2["h2_passed"] is True

        res = report["results_matrix"]
        assert "SBDS_LEARNED_GATE" in res
        assert "CONTRADICTORY_HISTORY" in res["SBDS_LEARNED_GATE"]
