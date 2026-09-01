"""Unit tests for SignerAdapter and AdaptedMultimodalModel (Prompt 41)."""

import pytest
import torch

from src.models.multimodal_baseline import MultimodalBaseline
from src.personalization.signer_adapter import (
    AdapterPlacement,
    SignerAdapter,
    AdaptedMultimodalModel,
)


def test_signer_adapter_forward_and_params():
    """Test bottleneck residual forward pass and parameter count < 50K."""
    adapter = SignerAdapter(in_dim=256, bottleneck_dim=16, placement=AdapterPlacement.TEMPORAL_REPRESENTATION)
    x = torch.randn(2, 10, 256)

    out = adapter(x)

    assert out.shape == x.shape
    params = adapter.count_parameters()
    # (256*16 + 16) + (16*256 + 256) = 4112 + 4352 = 8464 params
    assert params < 50000
    assert adapter.measure_memory_footprint_kb() > 0.0


def test_adapted_multimodal_model_frozen_base():
    """Verify base model parameters are frozen and only adapter parameters are trainable."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=256)
    adapted_model = AdaptedMultimodalModel(base_model=base_model, bottleneck_dim=16)

    # Check base model frozen
    for p in base_model.parameters():
        assert p.requires_grad is False

    # Check adapter trainable
    for p in adapted_model.adapter.parameters():
        assert p.requires_grad is True

    # Forward pass smoke test
    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)
    out = adapted_model(hands=hands, pose=pose)

    assert "logits" in out
