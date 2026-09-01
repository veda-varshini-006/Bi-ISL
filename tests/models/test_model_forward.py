"""Smoke tests for model forward pass."""

import pytest
import torch
import torch.nn as nn
from src.models.baseline_slt import BaseSLTModel

class DummySLTModel(nn.Module):
    """Dummy PyTorch SLT Model for forward pass smoke testing."""
    def __init__(self, in_features=128, hidden_dim=256, vocab_size=1000):
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, visual_inputs, context_inputs=None):
        # visual_inputs: (batch_size, seq_len, in_features)
        h = self.encoder(visual_inputs)
        logits = self.decoder(h)
        return {"logits": logits, "encoder_hidden": h}

def test_model_forward_pass_smoke():
    batch_size = 4
    seq_len = 30
    in_features = 128
    vocab_size = 1000

    model = DummySLTModel(in_features=in_features, vocab_size=vocab_size)
    dummy_input = torch.randn(batch_size, seq_len, in_features)

    outputs = model(dummy_input)

    assert "logits" in outputs
    assert "encoder_hidden" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, vocab_size)
    assert outputs["encoder_hidden"].shape == (batch_size, seq_len, 256)
