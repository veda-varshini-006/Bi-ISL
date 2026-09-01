"""Adaptation Consistency Score q_t Module (Prompt 43).

Calculates multi-view decoding consistency q_t across:
1. Temporal perturbation (frame stride / offset jitter)
2. Frame subsampling (50% frame dropout)
3. Mild spatial landmark augmentation (gaussian noise)
4. Token-level agreement across stochastic decodings.

Output:
Scalar consensus score q_t in [0.0, 1.0].
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptationConsistencyScorer:
    """Multi-view consistency metric q_t for UGSA adaptation gating."""

    def __init__(self, spatial_std: float = 0.01, frame_subsample_ratio: float = 0.8):
        self.spatial_std = spatial_std
        self.frame_subsample_ratio = frame_subsample_ratio

    def apply_spatial_augmentation(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Adds mild Gaussian noise to 3D landmarks."""
        noise = torch.randn_like(landmarks) * self.spatial_std
        return landmarks + noise

    def apply_frame_subsampling(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Subsamples frames along sequence dimension (dim=1)."""
        if landmarks.dim() < 2 or landmarks.shape[1] <= 4:
            return landmarks
        seq_len = landmarks.shape[1]
        keep_len = max(2, int(seq_len * self.frame_subsample_ratio))
        indices = torch.linspace(0, seq_len - 1, keep_len).long()
        return landmarks[:, indices, ...]

    def compute_token_agreement(self, seq_a: List[int], seq_b: List[int]) -> float:
        """Computes token-level Jaccard agreement between two token sequences."""
        if not seq_a or not seq_b:
            return 1.0 if seq_a == seq_b else 0.0
        set_a = set(seq_a)
        set_b = set(seq_b)
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / float(len(union)) if union else 1.0

    def calculate_consistency_score(
        self,
        base_logits: torch.Tensor,
        augmented_logits_list: List[torch.Tensor]
    ) -> float:
        """Calculates scalar q_t consistency score across model views using Cosine Similarity."""
        if not augmented_logits_list:
            return 1.0

        base_vec = F.softmax(base_logits.mean(dim=1), dim=-1)
        sims = []

        for aug_logits in augmented_logits_list:
            aug_vec = F.softmax(aug_logits.mean(dim=1), dim=-1)
            sim = F.cosine_similarity(base_vec, aug_vec, dim=-1).mean().item()
            sims.append(max(0.0, float(sim)))

        return float(sum(sims) / len(sims))
