"""Unit tests for ReverseISLEvaluator (Prompt 69)."""

import os
import tempfile
import pytest

from src.english_to_isl.reverse_evaluator import ReverseISLEvaluator


def test_evaluate_output_perfect_match():
    """Test evaluating perfect prediction match against reference."""
    evaluator = ReverseISLEvaluator()

    ref = {
        "intent": "ont:intent/symptom_report",
        "ordered_gloss_ids": ["TODAY", "FEVER", "HAVE"],
        "non_manual_markers": [{"gloss_id": "FEVER", "marker": "head_nod_slight"}],
        "preserves_isl_grammar": True,
        "is_random_nearest_substituted": False
    }

    scores = evaluator.evaluate_output(ref, ref)

    assert scores["intent_preservation"] == 1.0
    assert scores["semantic_correctness"] == 1.0
    assert scores["isl_ordering_correctness"] == 1.0
    assert scores["non_manual_marker_correctness"] == 1.0
    assert scores["unsupported_oov_handling"] == 1.0
    assert scores["overall_reverse_quality_score"] == 1.0


def test_evaluate_output_ordering_failure():
    """Test evaluating naive baseline output lacking ISL grammar ordering."""
    evaluator = ReverseISLEvaluator()

    ref = {
        "intent": "ont:intent/symptom_report",
        "ordered_gloss_ids": ["TODAY", "FEVER", "HAVE"],
        "non_manual_markers": [{"gloss_id": "FEVER", "marker": "head_nod_slight"}],
        "preserves_isl_grammar": True,
        "is_random_nearest_substituted": False
    }

    pred_naive = {
        "intent": "ont:intent/symptom_report",
        "ordered_gloss_ids": ["ME", "HAVE", "FEVER", "TODAY"],
        "non_manual_markers": [],
        "preserves_isl_grammar": False,
        "is_random_nearest_substituted": False
    }

    scores = evaluator.evaluate_output(ref, pred_naive)

    assert scores["isl_ordering_correctness"] == 0.0
    assert scores["non_manual_marker_correctness"] == 0.0
    assert scores["overall_reverse_quality_score"] < 0.700


def test_human_evaluation_rubric():
    """Test generating expert ISL-competent human evaluation guidelines."""
    evaluator = ReverseISLEvaluator()
    rubric = evaluator.generate_human_evaluation_rubric()

    assert "ISL-competent" in rubric["evaluator_qualification"]
    assert "facial_expression_naturalness" in rubric["qualitative_dimensions"]


def test_export_evaluation_report():
    """Test exporting JSON and MD report files."""
    evaluator = ReverseISLEvaluator()

    sample = {
        "intent_preservation": 1.0,
        "semantic_correctness": 1.0,
        "isl_ordering_correctness": 1.0,
        "non_manual_marker_correctness": 1.0,
        "unsupported_oov_handling": 1.0,
        "overall_reverse_quality_score": 1.0
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, data = evaluator.export_evaluation_report([sample])

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert data["summary"]["overall_quality_score"] == 1.0


def test_documentation_file_exists():
    """Verify REVERSE_EVALUATOR_SPEC.md exists."""
    doc_path = "./docs/evaluation/REVERSE_EVALUATOR_SPEC.md"
    assert os.path.exists(doc_path)
