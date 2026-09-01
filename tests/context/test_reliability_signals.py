"""Unit tests for Context Reliability Signals Estimator (Prompt 34)."""

import os
import tempfile
import pytest

from src.context.sbds_schema import (
    SharedBidirectionalDialogueState,
    Entity,
    EntityType,
    Intent,
    Referent,
    ReferentStatus,
)
from src.context.reliability_signals import ContextReliabilityEstimator, ReliabilitySignals


def test_reliability_signals_estimation_all_9_candidate_signals():
    """Test estimating all 9 required candidate reliability signals."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        estimator = ContextReliabilityEstimator(log_dir=tmp_dir)

        state = SharedBidirectionalDialogueState(
            sequence_id="seq_rel_1",
            active_entities=[Entity(entity_id="e1", name="Doctor", entity_type=EntityType.PERSON)],
            dialogue_intent=Intent(intent_type="MEDICAL_APPOINTMENT", slot_values={"room": "Room 101"}),
            unresolved_referents=[Referent(referent_id="r1", textual_anchor="he", status=ReferentStatus.UNRESOLVED)]
        )

        observation = {
            "english_text": "Doctor Sharma is in Room 202",
            "entities": [{"entity_id": "e1"}],
            "intent": {"intent_type": "MEDICAL_APPOINTMENT", "slot_values": {"room": "Room 202"}}
        }

        signals = estimator.estimate(
            state=state,
            current_observation=observation,
            visual_confidence=0.88,
            current_turn=3
        )

        assert isinstance(signals, ReliabilitySignals)
        assert signals.visual_confidence == 0.88
        assert signals.context_age_seconds > 0.0
        assert signals.entity_overlap_score > 0.0
        assert signals.intent_compatibility == 1.0
        assert signals.semantic_similarity >= 0.0
        assert signals.contradiction_indicator == 0.8  # Room 101 vs Room 202 contradiction
        assert signals.num_unresolved_referents == 1
        assert signals.context_confidence == 1.0
        assert signals.turn_distance == 3
        assert 0.0 <= signals.overall_reliability_score <= 1.0


def test_reliability_signals_logging_to_jsonl():
    """Test logging explainability signals to reliability_signals.jsonl file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        estimator = ContextReliabilityEstimator(log_dir=tmp_dir)
        state = SharedBidirectionalDialogueState(sequence_id="seq_rel_2")

        signals = estimator.estimate(state=state, current_observation={})

        jsonl_path = os.path.join(tmp_dir, "reliability_signals.jsonl")
        assert os.path.exists(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert "overall_reliability_score" in lines[0]
