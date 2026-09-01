"""Unit tests for DomainShiftEvaluator (Prompt 54)."""

import os
import tempfile
import pytest
import torch

from src.eval.domain_shift_evaluator import DomainShiftEvaluator


def test_apply_synthetic_shift_perturbation():
    """Test applying synthetic shift perturbations to 3D landmark tensor."""
    evaluator = DomainShiftEvaluator()
    landmarks = torch.randn(2, 16, 126)

    shifted = evaluator.apply_synthetic_shift(landmarks, shift_type="LIGHTING_CONTRAST", severity=0.5)
    assert shifted.shape == landmarks.shape
    assert not torch.allclose(shifted, landmarks)

    warped = evaluator.apply_synthetic_shift(landmarks, shift_type="SIGNING_SPEED_WARP", severity=0.8)
    assert warped.shape[1] <= landmarks.shape[1]


def test_evaluate_domain_shifts_isolation():
    """Test evaluating natural vs synthetic domain shift benchmark."""
    evaluator = DomainShiftEvaluator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, summary = evaluator.evaluate_domain_shifts(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert "natural_cross_dataset_shifts" in summary
        assert "synthetic_corruption_shifts" in summary

        # Methodological separation assertion
        assert summary["natural_cross_dataset_shifts"]["INCLUDE_Dataset"]["domain_gap_bleu"] < 0
        assert summary["synthetic_corruption_shifts"]["LIGHTING_CONTRAST"]["drop_bleu"] < 0


def test_documentation_file_exists():
    """Verify DOMAIN_SHIFT_EVALUATION_SPEC.md exists."""
    doc_path = "./docs/evaluation/DOMAIN_SHIFT_EVALUATION_SPEC.md"
    assert os.path.exists(doc_path)
