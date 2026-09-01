"""Unit tests for NaiveReverseBaseline (Prompt 68)."""

import os
import pytest

from src.english_to_isl.reverse_baseline import NaiveReverseBaseline


def test_naive_reverse_baseline_translation():
    """Test naive surface lookup translation."""
    baseline = NaiveReverseBaseline()

    res = baseline.translate_text("Where is the pharmacy?")

    assert res["baseline_name"] == "NaiveLookupBaseline"
    assert "WHERE" in res["ordered_gloss_ids"]
    assert "PHARMACY" in res["ordered_gloss_ids"]


def test_naive_baseline_flags():
    """Verify naive baseline flags reflect lack of grammar, semantics, and facial markers."""
    baseline = NaiveReverseBaseline()

    res = baseline.translate_text("I have a fever today.")

    assert res["has_non_manual_markers"] is False
    assert res["is_structured_semantic"] is False
    assert res["preserves_isl_grammar"] is False


def test_documentation_file_exists():
    """Verify NAIVE_REVERSE_BASELINE_SPEC.md exists."""
    doc_path = "./docs/baselines/NAIVE_REVERSE_BASELINE_SPEC.md"
    assert os.path.exists(doc_path)
