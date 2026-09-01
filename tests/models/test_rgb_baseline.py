"""Unit tests for Bi-ISL Learned Visual Encoder RGB/Video Baseline Architecture."""

import os
import tempfile
import pytest
import torch

from src.models.rgb_baseline import FrameEncoder, TemporalEncoder, TranslationDecoder, RGBVideoBaseline


def test_rgb_baseline_components_forward_pass():
    """Test component-level forward passes for FrameEncoder, TemporalEncoder, and TranslationDecoder."""
    # 1. FrameEncoder
    frame_enc = FrameEncoder(feature_dim=128, pretrained=False)
    dummy_rgb = torch.randn(2, 4, 3, 112, 112)
    spatial_feats = frame_enc(dummy_rgb)
    assert spatial_feats.shape == (2, 4, 128)

    # 2. TemporalEncoder
    temp_enc = TemporalEncoder(in_dim=128, hidden_dim=128, num_layers=2)
    temporal_feats = temp_enc(spatial_feats)
    assert temporal_feats.shape == (2, 4, 128)

    # 3. TranslationDecoder
    decoder = TranslationDecoder(hidden_dim=128, vocab_size=50)
    logits = decoder(temporal_feats)
    assert logits.shape == (2, 4, 50)


def test_rgb_baseline_full_forward_and_masking():
    """Test complete RGBVideoBaseline forward pass with attention masking."""
    model = RGBVideoBaseline(feature_dim=128, hidden_dim=128, vocab_size=50, pretrained=False)
    dummy_rgb = torch.randn(2, 8, 3, 112, 112)
    dummy_mask = torch.ones(2, 8, dtype=torch.bool)
    dummy_mask[1, 5:] = False

    logits = model(dummy_rgb, attention_mask=dummy_mask)
    assert logits.shape == (2, 8, 50)
    # Masked steps should be 0.0
    assert torch.all(logits[1, 5:] == 0.0)


def test_parameter_count_breakdown():
    """Test parameter count breakdown per component."""
    model = RGBVideoBaseline(feature_dim=128, hidden_dim=128, vocab_size=50, pretrained=False)
    params = model.count_parameters()

    assert "frame_encoder_parameters" in params
    assert "temporal_encoder_parameters" in params
    assert "translation_decoder_parameters" in params
    assert params["total_parameters"] > 0


def test_architecture_documentation_file_exists():
    """Test RGB_BASELINE_ARCHITECTURE.md documentation exists and covers components."""
    doc_path = "./docs/models/RGB_BASELINE_ARCHITECTURE.md"
    assert os.path.exists(doc_path)

    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    assert "Bi-ISL Learned Visual Encoder RGB/Video Baseline Architecture" in doc_text
    assert "Frame/Video Encoder" in doc_text
    assert "Temporal Modeling" in doc_text
    assert "Translation Decoder" in doc_text
    assert "SBDS" in doc_text
    assert "UGSA" in doc_text


def test_run_rgb_baseline_experiment():
    """Test running RGB baseline experiment runner generating performance reports."""
    from src.experiments.e1_rgb_baseline import run_rgb_baseline_experiment
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_rep, md_rep = run_rgb_baseline_experiment(epochs=1, batch_size=2, output_dir=tmp_dir)

        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            text = f.read()

        assert "RGB/Video Baseline Performance & Parameter Report" in text
        assert "Frame Encoder" in text
        assert "EXCLUDED (Strict Baseline)" in text
