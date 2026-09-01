"""Unit tests for Shared Bidirectional Dialogue State (SBDS) Schema (Prompt 31)."""

import json
import pytest
from pydantic import ValidationError

from src.context.sbds_schema import (
    CommunicationDirection,
    EntityType,
    Entity,
    Intent,
    ReferentStatus,
    Referent,
    TemporalContext,
    LocationContext,
    StateConfidence,
    ConfirmedTranslation,
    TurnMetadata,
    SharedBidirectionalDialogueState,
    SBDSReplayBuffer,
)


def test_sbds_instantiation_and_fields():
    """Test instantiating SBDS state object with all 8 sub-schemas."""
    entity = Entity(entity_id="e1", name="Doctor", entity_type=EntityType.PERSON, confidence=0.95)
    intent = Intent(intent_type="MEDICAL_QUERY", domain="HEALTHCARE", confidence=0.90)
    referent = Referent(referent_id="r1", textual_anchor="he", candidate_entity_ids=["e1"])
    temporal = TemporalContext(time_frame="PAST", tense="PAST")
    location = LocationContext(location_name="Hospital", room_type="CLINIC")
    confidence = StateConfidence(overall_confidence=0.92)
    translation = ConfirmedTranslation(turn_id="t1", speaker_id="s1", isl_glosses=["DOCTOR", "WHERE"], english_text="Where is the doctor?")
    turn = TurnMetadata(turn_index=1, active_speaker="USER_SIGNER", direction=CommunicationDirection.ISL_TO_ENGLISH)

    state = SharedBidirectionalDialogueState(
        sequence_id="seq_100",
        turn_metadata=turn,
        active_entities=[entity],
        dialogue_intent=intent,
        unresolved_referents=[referent],
        temporal_attributes=temporal,
        location_attributes=location,
        confidence_metadata=confidence,
        last_confirmed_translation=translation
    )

    assert state.sequence_id == "seq_100"
    assert len(state.active_entities) == 1
    assert state.active_entities[0].name == "Doctor"
    assert state.dialogue_intent.domain == "HEALTHCARE"
    assert state.unresolved_referents[0].referent_id == "r1"
    assert state.temporal_attributes.tense == "PAST"
    assert state.location_attributes.location_name == "Hospital"
    assert state.confidence_metadata.overall_confidence == 0.92
    assert state.last_confirmed_translation.english_text == "Where is the doctor?"


def test_sbds_serialization_and_deserialization():
    """Test JSON/dict serialization, deserialization, and SHA-256 state hash matching."""
    state = SharedBidirectionalDialogueState(sequence_id="seq_200")
    state_dict = state.to_dict()
    assert "state_hash" in state_dict
    assert len(state_dict["state_hash"]) == 64

    json_str = state.to_json()
    reconstructed = SharedBidirectionalDialogueState.from_json(json_str)

    assert reconstructed.sequence_id == state.sequence_id
    assert reconstructed.compute_state_hash() == state.compute_state_hash()


def test_sbds_immutability_and_next_version():
    """Test state immutability and create_next_version method."""
    state = SharedBidirectionalDialogueState(sequence_id="seq_300")

    # Immutability test (assignment to frozen model should raise ValidationError)
    with pytest.raises(ValidationError):
        state.sequence_id = "seq_400"  # type: ignore

    next_state = state.create_next_version(sequence_id="seq_300_v2")
    assert next_state.sequence_id == "seq_300_v2"
    assert next_state.compute_state_hash() != state.compute_state_hash()


def test_sbds_replay_buffer():
    """Test SBDSReplayBuffer trajectory storage and JSON round-trip."""
    buf = SBDSReplayBuffer()
    s1 = SharedBidirectionalDialogueState(sequence_id="seq_1")
    s2 = s1.create_next_version(sequence_id="seq_2")

    buf.append(s1)
    buf.append(s2)

    assert len(buf.get_trajectory()) == 2

    buf_json = buf.to_json()
    reloaded_buf = SBDSReplayBuffer.from_json(buf_json)
    assert len(reloaded_buf.get_trajectory()) == 2
    assert reloaded_buf.get_trajectory()[0].sequence_id == "seq_1"


def test_no_arbitrary_full_transcript_as_primary_state():
    """Assert primary state is structured schema without arbitrary full transcript field."""
    state = SharedBidirectionalDialogueState()
    d = state.to_dict()
    assert "full_transcript" not in d
    assert "arbitrary_transcript" not in d
