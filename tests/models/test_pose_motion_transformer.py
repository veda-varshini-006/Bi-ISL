"""Tests for Pose-Motion Transformer translation architecture & BiISLTranslator."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.pose_motion_transformer import (
    PoseMotionEncoder,
    PoseMotionTranslationModel,
    BiISLTranslator
)


def test_pose_motion_encoder_forward():
    encoder = PoseMotionEncoder(d_model=256, dropout=0.0)
    encoder.eval()

    # Input (B, T, 75, 3)
    x = torch.randn(2, 128, 75, 3)
    out = encoder(x)
    assert out.shape == (2, 128, 256)

    # Input flattened 225
    x_flat = torch.randn(2, 128, 225)
    out_flat = encoder(x_flat)
    assert out_flat.shape == (2, 128, 256)


def test_pose_motion_translation_model_forward():
    model = PoseMotionTranslationModel(vocab_size=5000, d_model=256)
    model.eval()

    pose = torch.randn(1, 128, 75, 3)
    decoder_in = torch.tensor([[2, 45, 89]], dtype=torch.long)  # (B, T_tgt)
    logits = model(pose, decoder_in)

    assert logits.shape == (1, 3, 5000)


def test_biisl_translator_load_default():
    translator = BiISLTranslator.load_default(device="cpu")
    assert translator.model is not None
    assert isinstance(translator.model, PoseMotionTranslationModel)


def test_biisl_translator_inference_shape():
    translator = BiISLTranslator(device="cpu")
    dummy_landmarks = np.random.randn(128, 75, 3).astype(np.float32)

    text, conf = translator.translate(dummy_landmarks, max_new_tokens=10, return_confidence=True)
    assert isinstance(text, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
