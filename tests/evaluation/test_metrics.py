"""Smoke tests for evaluation metric computation."""

import pytest
from src.metrics.translation_metrics import BaseTranslationMetric
from src.metrics.context_metrics import BaseContextMetric

class DummyTranslationMetric(BaseTranslationMetric):
    def compute_score(self, hypotheses, references):
        # Dummy compute BLEU-4 and chrF
        return {
            "bleu4": 25.0,
            "chrf": 42.0,
            "bertscore": 0.88
        }

class DummyContextMetric(BaseContextMetric):
    def compute_unsupported_slot_rate(self, hypotheses, visual_ground_truth):
        return 0.02

def test_translation_metric_smoke():
    metric = DummyTranslationMetric()
    hyps = ["hello world", "thank you very much"]
    refs = ["hello world", "thank you so much"]
    scores = metric.compute_score(hyps, refs)

    assert "bleu4" in scores
    assert "chrf" in scores
    assert scores["bleu4"] == 25.0

def test_context_usr_metric_smoke():
    metric = DummyContextMetric()
    usr = metric.compute_unsupported_slot_rate(["hyps"], [{"slots": []}])
    assert usr == 0.02
