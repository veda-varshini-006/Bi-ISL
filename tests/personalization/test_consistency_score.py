"""Unit tests for AdaptationConsistencyScorer (Prompt 43)."""

import pytest
import torch

from src.personalization.consistency_score import AdaptationConsistencyScorer


def test_adaptation_consistency_scorer_methods():
    """Test spatial augmentation, frame subsampling, and cosine similarity q_t."""
    scorer = AdaptationConsistencyScorer()

    landmarks = torch.randn(2, 16, 126)
    aug_landmarks = scorer.apply_spatial_augmentation(landmarks)

    assert aug_landmarks.shape == landmarks.shape
    assert not torch.allclose(aug_landmarks, landmarks)

    subsampled = scorer.apply_frame_subsampling(landmarks)
    assert subsampled.shape[1] < landmarks.shape[1]

    # Token agreement
    seq_a = [1, 2, 3, 4]
    seq_b = [1, 2, 3, 5]
    agree = scorer.compute_token_agreement(seq_a, seq_b)

    assert 0.0 < agree < 1.0

    # Consistency score q_t
    base_logits = torch.randn(2, 10, 20)
    aug_logits_1 = base_logits + torch.randn_like(base_logits) * 0.05
    aug_logits_2 = base_logits + torch.randn_like(base_logits) * 0.10

    q_t = scorer.calculate_consistency_score(base_logits, [aug_logits_1, aug_logits_2])

    assert 0.0 <= q_t <= 1.0
