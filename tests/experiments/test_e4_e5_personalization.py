"""Unit tests for E4/E5 Personalization & Adaptation-Noise Benchmark (Prompt 50)."""

import os
import tempfile
import pytest

from src.experiments.e4_e5_personalization import run_e4_e5_personalization_experiment


def test_run_e4_e5_personalization_experiment():
    """Test running E4/E5 personalization experiment evaluating 4 systems, 10 signers, and Hypothesis H3."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, report = run_e4_e5_personalization_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert len(report["signers"]) == 10
        assert len(report["system_summaries"]) == 4

        h3 = report["hypothesis_testing"]["H3_ugsa_personalization_safety"]
        assert h3["h3_passed"] is True

        summaries = report["system_summaries"]
        assert summaries["PROPOSED_UGSA"]["mean_signer_gain_clean"] > 3.0
        assert summaries["PROPOSED_UGSA"]["worst_signer_degradation_clean"] == 0.0
        assert summaries["NAIVE_ADAPTATION_BASELINE"]["worst_signer_degradation_noisy"] < 0.0
