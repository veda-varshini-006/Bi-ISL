"""Unit tests for Bi-ISL Experiment E1 Evaluator."""

import os
import tempfile
import pytest

from src.evaluation.e1_evaluator import E1Evaluator


def test_bleu_and_chrf_computation():
    """Test BLEU-1 to BLEU-4 and chrF++ metrics calculation."""
    evaluator = E1Evaluator()
    preds = ["hello world sign language", "today is Tuesday"]
    refs = ["hello world sign language", "today is Wednesday"]

    bleu = evaluator.compute_bleu(preds, refs)
    chrf = evaluator.compute_chrf(preds, refs)

    assert "bleu_1" in bleu
    assert "bleu_4" in bleu
    assert bleu["bleu_1"] > 0.0
    assert chrf > 0.0


def test_wer_computation():
    """Test Word Error Rate (WER) with substitutions, insertions, deletions."""
    evaluator = E1Evaluator()
    res = evaluator.compute_wer("hello world sign", "hello world sign language")
    assert res["wer"] == 25.0  # 1 deletion out of 4 words = 25%
    assert res["edit_distance"] == 1


def test_semantic_score_v1():
    """Test version-pinned semantic similarity metric (v1.0.0)."""
    evaluator = E1Evaluator()
    sem = evaluator.compute_semantic_score_v1("hello world sign language", "hello world sign language")
    assert sem == 100.0


def test_e1_evaluator_artifact_generation():
    """Test full evaluation pipeline generating all 4 required files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = E1Evaluator()
        sids = ["sample_1", "sample_2"]
        preds = ["good morning everyone", "thank you very much"]
        refs = ["good morning everyone", "thank you so much"]

        metrics = evaluator.evaluate(sids, preds, refs, output_dir=tmp_dir)

        assert os.path.exists(os.path.join(tmp_dir, "predictions.jsonl"))
        assert os.path.exists(os.path.join(tmp_dir, "references.jsonl"))
        assert os.path.exists(os.path.join(tmp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(tmp_dir, "scores.csv"))

        assert "bleu_1" in metrics
        assert "chrf_plus_plus" in metrics
        assert "length_statistics" in metrics


def test_run_e1_evaluation_experiment_runner():
    """Test running E1 evaluation experiment runner script."""
    from src.experiments.e1_evaluation_runner import run_e1_evaluation_experiment

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir, metrics = run_e1_evaluation_experiment(output_dir=tmp_dir)
        assert os.path.exists(os.path.join(out_dir, "predictions.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "references.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "metrics.json"))
        assert os.path.exists(os.path.join(out_dir, "scores.csv"))
        assert "bleu_4" in metrics
        assert "mean_wer" in metrics
