"""Unit tests for Bi-ISL Multimodal Baseline Model and E2 Ablation Runner."""

import os
import tempfile
import pytest
import torch

from src.models.multimodal_baseline import MultimodalBaseline
from src.experiments.e2_multimodal_ablation import run_multimodal_ablation_experiment


def test_multimodal_baseline_forward_all_5_ablation_configs():
    """Test forward pass across all 5 modality ablation configurations."""
    model = MultimodalBaseline(
        rgb_feature_dim=64,
        hand_dim=126,
        pose_dim=132,
        face_dim=1404,
        fusion_dim=128,
        vocab_size=30
    )

    dummy_rgb = torch.randn(2, 8, 3, 112, 112)
    dummy_hands = torch.randn(2, 8, 126)
    dummy_pose = torch.randn(2, 8, 132)
    dummy_face = torch.randn(2, 8, 1404)
    dummy_mask = torch.ones(2, 8, dtype=torch.bool)

    ablation_configs = [
        ("RGB_ONLY", True, False, False, False),
        ("LANDMARKS_ONLY", False, True, True, True),
        ("RGB_HANDS", True, True, False, False),
        ("RGB_HANDS_POSE", True, True, True, False),
        ("RGB_HANDS_POSE_FACE", True, True, True, True)
    ]

    for name, u_rgb, u_h, u_p, u_f in ablation_configs:
        logits = model(
            rgb=dummy_rgb,
            hands=dummy_hands,
            pose=dummy_pose,
            face=dummy_face,
            use_rgb=u_rgb,
            use_hands=u_h,
            use_pose=u_p,
            use_face=u_f,
            attention_mask=dummy_mask
        )
        assert logits.shape == (2, 8, 30)


def test_explicit_modality_zero_masking():
    """Test explicit modality zero-out masking behavior."""
    model = MultimodalBaseline(fusion_dim=64, vocab_size=20)
    dummy_hands = torch.randn(2, 4, 126)

    logits_none = model(hands=dummy_hands, use_rgb=False, use_hands=False, use_pose=False, use_face=False)
    assert logits_none.shape == (2, 4, 20)


def test_e2_multimodal_ablation_experiment_runner():
    """Test running E2 ablation study generating JSON and Markdown reports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_rep, md_rep = run_multimodal_ablation_experiment(batch_size=2, seq_len=8, output_dir=tmp_dir)

        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            md_text = f.read()

        assert "Experiment E2: Multimodal Modality Ablation Study Report" in md_text
        assert "RGB_ONLY" in md_text
        assert "RGB_HANDS_POSE_FACE" in md_text
