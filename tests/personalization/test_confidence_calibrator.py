"""Unit tests for ConfidenceCalibrator (Prompt 42)."""

import os
import pytest
import numpy as np
import torch

from src.personalization.confidence_calibrator import TemperatureScaler, ConfidenceCalibrator


def test_temperature_scaling_calibrator():
    """Test temperature scaling logits and softmax probability output."""
    calibrator = ConfidenceCalibrator(default_temperature=2.0)
    logits = torch.tensor([[5.0, 1.0, 0.0]])

    cal_probs = calibrator.calibrate_logits(logits)
    raw_probs = torch.softmax(logits, dim=-1)

    assert cal_probs.shape == raw_probs.shape
    # Higher temperature softens probabilities (max prob is smaller)
    assert cal_probs.max() < raw_probs.max()


def test_entropy_and_sequence_agreement():
    """Test predictive entropy and sequence agreement calculation."""
    calibrator = ConfidenceCalibrator()

    # High entropy (uniform distribution)
    uniform_p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    ent_high = calibrator.compute_predictive_entropy(uniform_p)

    # Low entropy (peaked distribution)
    peaked_p = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    ent_low = calibrator.compute_predictive_entropy(peaked_p)

    assert ent_high.item() > ent_low.item()

    # Sequence agreement
    samples = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 4]]
    agree = calibrator.compute_sequence_agreement(samples)
    assert agree == 2.0 / 3.0


def test_ece_and_brier_score():
    """Test Expected Calibration Error (ECE) and Brier Score computation."""
    calibrator = ConfidenceCalibrator(num_bins=5)

    confs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    accs = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    ece, bins_data = calibrator.compute_ece(confs, accs)
    assert 0.0 <= ece <= 1.0
    assert len(bins_data) > 0

    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    brier = calibrator.compute_brier_score(probs, targets)

    assert brier >= 0.0


def test_documentation_file_exists():
    """Verify CONFIDENCE_CALIBRATION_SPEC.md exists."""
    doc_path = "./docs/personalization/CONFIDENCE_CALIBRATION_SPEC.md"
    assert os.path.exists(doc_path)
