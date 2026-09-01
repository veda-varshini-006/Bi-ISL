"""Unit tests for Bi-ISL Official Baseline Freeze Subsystem."""

import os
import tempfile
import pytest

from src.experiments.e1_baseline_freeze import run_baseline_freeze_suite, get_git_commit_hash


def test_get_git_commit_hash():
    """Test retrieving current git commit SHA."""
    commit_sha = get_git_commit_hash()
    assert isinstance(commit_sha, str)
    assert len(commit_sha) > 0


def test_run_baseline_freeze_suite_multi_seed():
    """Test executing baseline freeze suite across 3 seeds and selecting strongest model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_rep, md_doc, freeze_data = run_baseline_freeze_suite(seeds=[42, 123, 456], output_dir=tmp_dir)

        assert os.path.exists(json_rep)
        assert os.path.exists(md_doc)

        assert freeze_data["tag"] == "BASELINE_V1"
        assert len(freeze_data["seeds_evaluated"]) == 3
        assert freeze_data["validation_bleu4_mean"] > 0.0

        # Check all 6 frozen components
        fc = freeze_data["frozen_components"]
        assert "configuration" in fc
        assert "splits" in fc
        assert "tokenizer" in fc
        assert "evaluation_code" in fc
        assert "checkpoint" in fc
        assert "commit_hash" in freeze_data


def test_baseline_freeze_document_generation():
    """Test BASELINE_FREEZE.md document contents."""
    doc_path = "./docs/baselines/BASELINE_FREEZE.md"
    assert os.path.exists(doc_path)

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Bi-ISL Official Baseline Freeze Document (BASELINE_V1)" in text
    assert "BASELINE_V1" in text
    assert "config/base_config.yaml" in text
    assert "src/evaluation/e1_evaluator.py" in text
    assert "src/text/tokenizer.py" in text
