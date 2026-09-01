"""Unit tests for 2x2 Factorial Ablation Experiment (Prompt 52)."""

import os
import tempfile
import pytest

from src.experiments.e6_factorial_ablation import run_2x2_factorial_ablation_experiment


def test_run_2x2_factorial_ablation_experiment():
    """Test executing 2x2 factorial ablation runner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, report = run_2x2_factorial_ablation_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        configs = report["configurations"]
        assert len(configs) == 4

        analysis = report["factorial_analysis"]
        assert analysis["main_effect_sbds_context"] > 0
        assert analysis["main_effect_ugsa_personalization"] > 0
        assert "interaction_effect_synergy" in analysis


def test_documentation_file_exists():
    """Verify FACTORIAL_ABLATION_ANALYSIS.md exists."""
    doc_path = "./docs/experiments/FACTORIAL_ABLATION_ANALYSIS.md"
    assert os.path.exists(doc_path)
