"""Unit tests for SBDS Context Encoder Subsystem (Prompt 33)."""

import os
import tempfile
import pytest
import torch

from src.context.sbds_schema import SharedBidirectionalDialogueState, Entity, EntityType, Intent
from src.context.context_encoder import SBDSContextEncoder, EncoderArchitecture
from src.experiments.e3_context_encoder_ablation import run_context_encoder_ablation_experiment


def test_simple_embedding_context_encoder():
    """Test SIMPLE_EMBEDDING context encoder forward pass and output shape."""
    encoder = SBDSContextEncoder(embed_dim=256, architecture=EncoderArchitecture.SIMPLE_EMBEDDING)
    states = [SharedBidirectionalDialogueState(sequence_id="s1"), SharedBidirectionalDialogueState(sequence_id="s2")]

    out, mask = encoder(states)

    assert out.shape == (2, 256)
    assert mask.shape[0] == 2
    assert mask.dtype == torch.bool


def test_transformer_attention_context_encoder():
    """Test TRANSFORMER_ATTENTION context encoder forward pass and output shape."""
    encoder = SBDSContextEncoder(embed_dim=256, architecture=EncoderArchitecture.TRANSFORMER_ATTENTION)
    states = [SharedBidirectionalDialogueState(sequence_id="s1")]

    out, mask = encoder(states)

    assert out.dim() == 3
    assert out.shape[0] == 1
    assert out.shape[2] == 256
    assert mask.dtype == torch.bool


def test_missing_state_component_masking():
    """Test component mask properly identifies valid vs missing state slots."""
    encoder = SBDSContextEncoder(embed_dim=128)

    state_with_entity = SharedBidirectionalDialogueState(
        active_entities=[Entity(entity_id="e1", name="Doctor", entity_type=EntityType.PERSON)]
    )
    state_empty = SharedBidirectionalDialogueState(active_entities=[])

    _, mask_valid = encoder([state_with_entity])
    _, mask_empty = encoder([state_empty])

    # Populated state should have more True positions in mask than empty state
    assert torch.sum(mask_valid).item() > torch.sum(mask_empty).item()


def test_run_context_encoder_ablation_experiment():
    """Test running context encoder ablation experiment generating report artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, res = run_context_encoder_ablation_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert "SIMPLE_EMBEDDING" in res
        assert "TRANSFORMER_ATTENTION" in res
        assert res["SIMPLE_EMBEDDING"]["parameter_count"] > 0
        assert res["TRANSFORMER_ATTENTION"]["parameter_count"] > 0
