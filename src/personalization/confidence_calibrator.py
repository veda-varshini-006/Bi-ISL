"""Confidence Calibration for Personalization & Adaptation Decisions (Prompt 42).

Implements post-hoc temperature scaling and uncertainty metrics for adaptation decisions:
- Maximum token/sequence confidence
- Predictive entropy H(p)
- Temperature scaling: p_T = softmax(logits / T)
- Sequence agreement rate across Monte Carlo samples
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagram binned confidence curves

STRICT RULE:
Never treat raw softmax confidence as automatically calibrated.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learned temperature parameter T > 0 for logit scaling."""

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        t = torch.clamp(self.temperature, min=0.01)
        return logits / t


class ConfidenceCalibrator:
    """Confidence calibration and uncertainty evaluator for adaptation gating."""

    def __init__(self, num_bins: int = 10, default_temperature: float = 1.5):
        self.num_bins = num_bins
        self.scaler = TemperatureScaler(init_temperature=default_temperature)

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to raw logits before softmax."""
        scaled_logits = self.scaler(logits)
        return F.softmax(scaled_logits, dim=-1)

    def compute_predictive_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Computes predictive entropy H(p) = -sum(p * log(p + eps))."""
        eps = 1e-8
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def compute_sequence_agreement(self, sample_predictions: List[List[int]]) -> float:
        """Computes sequence agreement rate across multiple stochastic decodings."""
        if not sample_predictions or len(sample_predictions) <= 1:
            return 1.0
        first_seq = sample_predictions[0]
        matches = sum(1 for seq in sample_predictions[1:] if seq == first_seq)
        return matches / float(len(sample_predictions) - 1)

    def compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Computes Expected Calibration Error (ECE) and reliability curve bins."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        bins_data = []

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = float(np.mean(accuracies[in_bin]))
                avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

                bins_data.append({
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "count": int(np.sum(in_bin)),
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin
                })

        return float(ece), bins_data

    def compute_brier_score(
        self,
        probs: np.ndarray,
        targets_onehot: np.ndarray
    ) -> float:
        """Computes Brier Score = (1/N) * sum((p_i - y_i)^2)."""
        return float(np.mean(np.sum((probs - targets_onehot) ** 2, axis=-1)))
