"""Unit tests for Fixed-Weight Context Fusion Baseline Experiment (Prompt 38)."""

import os
import tempfile
import pytest

from src.experiments.e3_fixed_weight_ablation import run_fixed_weight_context_experiment


def test_fixed_weight_context_experiment_runner():
    """Test evaluating predefined fixed alpha values vs learned gating on validation set."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, doc_p, summary = run_fixed_weight_context_experiment(
            fixed_alphas=[0.0, 0.5, 1.0],
            output_dir=tmp_dir
        )

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert os.path.exists(doc_p)

        results = summary["results"]
        assert "fixed_alpha_0.00" in results
        assert "fixed_alpha_0.50" in results
        assert "fixed_alpha_1.00" in results
        assert "learned_reliability_gate" in results

        assert summary["best_fixed_bleu4"] > 0.0
        assert "learned_vs_best_fixed_delta" in summary


def test_documentation_and_report_file_exists():
    """Test FIXED_WEIGHT_CONTEXT_ABLATION.md document exists in docs/baselines/."""
    doc_path = "./docs/baselines/FIXED_WEIGHT_CONTEXT_ABLATION.md"
    assert os.path.exists(doc_path)
