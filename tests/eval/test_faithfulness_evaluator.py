"""Unit tests for SemanticFaithfulnessEvaluator (Prompt 57)."""

import os
import tempfile
import pytest

from src.eval.faithfulness_evaluator import SemanticFaithfulnessEvaluator


def test_extract_slots_all_categories():
    """Test slot extraction for all 8 categories."""
    evaluator = SemanticFaithfulnessEvaluator()
    sample_text = "Where is doctor in hospital for fever on left today with no pain?"

    slots = evaluator.extract_slots(sample_text)

    assert "doctor" in slots["entity"]
    assert "hospital" in slots["location"]
    assert "fever" in slots["symptom_object"]
    assert "left" in slots["direction"]
    assert "today" in slots["time"]
    assert "no" in slots["negation"]
    assert "where" in slots["question_type"]


def test_evaluate_sentence_pair_precision_recall():
    """Test slot precision, recall, and F1 calculation on reference and hypothesis pair."""
    evaluator = SemanticFaithfulnessEvaluator()

    ref = "Doctor is in hospital for fever today."
    hyp = "Doctor is in clinic for fever today."

    eval_res = evaluator.evaluate_sentence_pair(ref, hyp)

    assert 0.0 <= eval_res["overall_precision"] <= 1.0
    assert 0.0 <= eval_res["overall_recall"] <= 1.0
    assert 0.0 <= eval_res["overall_f1"] <= 1.0


def test_evaluate_corpus_metrics():
    """Test corpus-wide evaluation alongside BLEU-4 and chrF metrics."""
    evaluator = SemanticFaithfulnessEvaluator()

    refs = [
        "Doctor is in hospital for fever today.",
        "Where is pharmacy on left side?"
    ]
    hyps = [
        "Doctor is in hospital for fever today.",
        "Where is clinic on left side?"
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, summary = evaluator.evaluate_corpus(
            references=refs,
            hypotheses=hyps,
            bleu_4=23.60,
            chrf=54.20,
            output_dir=tmp_dir
        )

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        metrics = summary["metrics"]
        assert metrics["bleu_4"] == 23.60
        assert metrics["chrf"] == 54.20
        assert metrics["semantic_slot_f1"] > 0.5


def test_documentation_file_exists():
    """Verify SEMANTIC_FAITHFULNESS_SPEC.md exists."""
    doc_path = "./docs/evaluation/SEMANTIC_FAITHFULNESS_SPEC.md"
    assert os.path.exists(doc_path)
