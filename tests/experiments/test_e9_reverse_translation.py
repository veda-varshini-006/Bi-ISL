"""Unit tests for Experiment E9 Reverse Translation (Prompt 70)."""

import os
import tempfile
import pytest

from src.experiments.e9_reverse_translation import run_e9_reverse_translation_experiment


def test_run_e9_reverse_translation_experiment():
    """Test running Experiment E9 comparing naive baseline vs structured generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, res = run_e9_reverse_translation_experiment(report_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert res["experiment"] == "E9_REVERSE_TRANSLATION_BENCHMARK"

        assert res["structured_semantic_generation"]["overall_quality_score"] > res["naive_lookup_baseline"]["overall_quality_score"]
        assert res["structured_semantic_generation"]["mean_isl_ordering_correctness"] > res["naive_lookup_baseline"]["mean_isl_ordering_correctness"]


def test_e9_6_category_error_taxonomy_breakdown():
    """Test 6-category error taxonomy breakdown in E9 report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, res = run_e9_reverse_translation_experiment(report_dir=tmp_dir)

        naive_errs = res["naive_lookup_baseline"]["error_taxonomy"]
        struct_errs = res["structured_semantic_generation"]["error_taxonomy"]

        for cat in ["semantic_loss", "ordering_error", "missing_nmm", "wrong_sign", "oov_failure", "timing_issue"]:
            assert cat in naive_errs
            assert cat in struct_errs

        assert struct_errs["ordering_error"] == 0.0
        assert struct_errs["missing_nmm"] == 0.0


def test_production_readiness_caveat_enforcement():
    """Verify production readiness is NOT claimed based solely on automated metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, res = run_e9_reverse_translation_experiment(report_dir=tmp_dir)

        assert res["production_readiness_status"] == "NOT_PRODUCTION_READY_AUTOMATED_ONLY"
        assert "cannot be declared production-ready" in res["production_readiness_caveat"]


def test_documentation_file_exists():
    """Verify E9_REVERSE_TRANSLATION_EXPERIMENT.md exists."""
    doc_path = "./docs/experiments/E9_REVERSE_TRANSLATION_EXPERIMENT.md"
    assert os.path.exists(doc_path)
