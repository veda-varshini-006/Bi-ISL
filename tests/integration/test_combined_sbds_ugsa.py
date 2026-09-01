"""Integration tests for CombinedSBDSUGSAPipeline (Prompt 51)."""

import os
import pytest
import torch

from src.models.multimodal_baseline import MultimodalBaseline
from src.context.sbds_schema import SharedBidirectionalDialogueState, Intent
from src.integration.combined_sbds_ugsa_pipeline import CombinedSBDSUGSAPipeline


def test_combined_pipeline_forward_both_enabled():
    """Test full forward pass with both SBDS context gating and UGSA adaptation active."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=256)
    pipeline = CombinedSBDSUGSAPipeline(base_model=base_model, enable_sbds=True, enable_ugsa=True)

    sbds_state = SharedBidirectionalDialogueState(
        sequence_id="seq_integration_1",
        dialogue_intent=Intent(intent_type="CONSULTATION", domain="MEDICAL")
    )

    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)

    out = pipeline(sbds_state=sbds_state, hands=hands, pose=pose)

    assert "logits" in out
    assert "p_t" in out
    assert "q_t" in out
    assert "alpha_t" in out
    assert out["enable_sbds"] is True
    assert out["enable_ugsa"] is True


def test_combined_pipeline_independent_toggles():
    """Test toggling enable_sbds and enable_ugsa independently."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=256)

    pipeline_no_sbds = CombinedSBDSUGSAPipeline(base_model=base_model, enable_sbds=False, enable_ugsa=True)
    pipeline_no_ugsa = CombinedSBDSUGSAPipeline(base_model=base_model, enable_sbds=True, enable_ugsa=False)

    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)

    out_no_sbds = pipeline_no_sbds(hands=hands, pose=pose)
    assert out_no_sbds["alpha_t"] == 0.0

    out_no_ugsa = pipeline_no_ugsa(hands=hands, pose=pose)
    assert out_no_ugsa["enable_ugsa"] is False


def test_context_gating_isolation_untainted_metrics():
    """Verify context gating does not modify visual adaptation confidence metrics (p_t, q_t)."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=256)
    pipeline = CombinedSBDSUGSAPipeline(base_model=base_model, enable_sbds=True, enable_ugsa=True)
    pipeline.eval()

    sbds_state = SharedBidirectionalDialogueState(
        sequence_id="seq_isolation",
        dialogue_intent=Intent(intent_type="EMERGENCY", domain="MEDICAL")
    )

    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)

    pipeline.enable_sbds = True
    out_sbds = pipeline(sbds_state=sbds_state, hands=hands, pose=pose)

    pipeline.enable_sbds = False
    out_plain = pipeline(hands=hands, pose=pose)

    # p_t & q_t must be identical regardless of SBDS context gating
    assert abs(out_sbds["p_t"] - out_plain["p_t"]) < 1e-5
    assert abs(out_sbds["q_t"] - out_plain["q_t"]) < 1e-5


def test_adaptation_isolation_immutable_state():
    """Verify online adapter updates do not modify SBDS dialogue state object or hash."""
    base_model = MultimodalBaseline(vocab_size=20, fusion_dim=256)
    pipeline = CombinedSBDSUGSAPipeline(base_model=base_model, enable_sbds=True, enable_ugsa=True)

    sbds_state = SharedBidirectionalDialogueState(
        sequence_id="seq_immutable",
        dialogue_intent=Intent(intent_type="CONSULTATION", domain="GENERAL")
    )

    hash_before = sbds_state.compute_state_hash()

    # Perform gradient update on adapter parameters
    optimizer = torch.optim.AdamW(pipeline.adapted_model.adapter.parameters(), lr=1e-3)
    hands = torch.randn(2, 16, 126)
    pose = torch.randn(2, 16, 132)

    out = pipeline(sbds_state=sbds_state, hands=hands, pose=pose)
    loss = torch.mean(out["logits"] ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    hash_after = sbds_state.compute_state_hash()

    # SBDS dialogue state hash remains 100% unchanged
    assert hash_before == hash_after


def test_documentation_file_exists():
    """Verify COMBINED_SBDS_UGSA_ARCHITECTURE.md exists."""
    doc_path = "./docs/integration/COMBINED_SBDS_UGSA_ARCHITECTURE.md"
    assert os.path.exists(doc_path)
