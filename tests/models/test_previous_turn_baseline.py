"""Unit tests for Ungated Previous-Turn Context Baseline (Prompt 37)."""

import os
import pytest
import torch

from src.models.multimodal_baseline import MultimodalBaseline
from src.models.previous_turn_baseline import PreviousTurnTextEncoder, PreviousTurnBaseline


def test_previous_turn_text_encoder():
    """Test text encoder converting previous token sequence into context vector."""
    text_enc = PreviousTurnTextEncoder(vocab_size=100, embed_dim=64)
    tokens = torch.randint(0, 100, (2, 10))

    c_prev = text_enc(tokens)

    assert c_prev.shape == (2, 64)


def test_previous_turn_baseline_forward_with_text():
    """Test forward pass of PreviousTurnBaseline fusing previous turn text."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=64)
    prev_baseline = PreviousTurnBaseline(base_visual_model=base_model, vocab_size=100, embed_dim=64)

    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)
    prev_tokens = torch.randint(0, 100, (2, 10))

    out = prev_baseline(prev_target_tokens=prev_tokens, hands=hands, pose=pose)

    assert "logits" in out
    assert out["logits"].dim() == 3
    assert out["logits"].shape[0] == 2
    assert out["logits"].shape[2] == 20


def test_previous_turn_baseline_no_sbds_or_gate():
    """Assert model contains no SBDS state or reliability gating logic."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=64)
    prev_baseline = PreviousTurnBaseline(base_visual_model=base_model, vocab_size=100, embed_dim=64)

    assert not hasattr(prev_baseline, "context_gate")
    assert not hasattr(prev_baseline, "sbds_schema")


def test_architecture_documentation_file_exists():
    """Test documentation file exists in docs/models/."""
    doc_path = "./docs/models/PREVIOUS_TURN_BASELINE_ARCHITECTURE.md"
    assert os.path.exists(doc_path)
