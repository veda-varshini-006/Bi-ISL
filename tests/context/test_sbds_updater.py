"""Unit tests for Shared Bidirectional Dialogue State (SBDS) Updater (Prompt 32)."""

import pytest

from src.context.sbds_schema import (
    CommunicationDirection,
    EntityType,
    Entity,
    Intent,
    ReferentStatus,
    Referent,
    SharedBidirectionalDialogueState,
    SBDSReplayBuffer,
)
from src.context.sbds_updater import SBDSUpdater


def test_entity_creation_and_update():
    """Test creating a new entity and subsequently updating its attributes."""
    updater = SBDSUpdater()
    state_0 = SharedBidirectionalDialogueState(sequence_id="seq_test_1")

    # Turn 1: Create Entity
    input_t1 = {
        "speaker_id": "SIGNER_1",
        "english_text": "Dr. Sharma arrives at the clinic.",
        "entities": [
            {"entity_id": "doc_1", "name": "Dr. Sharma", "entity_type": "PERSON", "confidence": 0.90, "attributes": {"role": "Doctor"}}
        ]
    }
    state_1 = updater.update(state_0, input_t1, confidence=0.95)
    assert len(state_1.active_entities) == 1
    assert state_1.active_entities[0].name == "Dr. Sharma"
    assert state_1.active_entities[0].attributes["role"] == "Doctor"
    assert state_1.turn_metadata.turn_index == 1

    # Turn 2: Update existing Entity attribute
    input_t2 = {
        "speaker_id": "SIGNER_1",
        "english_text": "Dr. Sharma is a neurologist.",
        "entities": [
            {"entity_id": "doc_1", "name": "Dr. Sharma", "entity_type": "PERSON", "confidence": 0.98, "attributes": {"specialty": "Neurology"}}
        ]
    }
    state_2 = updater.update(state_1, input_t2, confidence=0.98)
    assert len(state_2.active_entities) == 1
    assert state_2.active_entities[0].attributes["role"] == "Doctor"
    assert state_2.active_entities[0].attributes["specialty"] == "Neurology"
    assert state_2.active_entities[0].confidence == 0.98


def test_pronoun_reference_tracking():
    """Test pronoun referent resolution linking 'he' to candidate entity."""
    updater = SBDSUpdater()
    state_0 = SharedBidirectionalDialogueState(sequence_id="seq_test_2")

    # Turn 1: Introduce entity
    input_t1 = {
        "speaker_id": "SIGNER_1",
        "entities": [{"entity_id": "patient_1", "name": "Rahul", "entity_type": "PERSON"}]
    }
    state_1 = updater.update(state_0, input_t1)

    # Turn 2: Introduce unresolved referent "he"
    input_t2 = {
        "speaker_id": "SIGNER_1",
        "english_text": "He needs medicine.",
        "referents": [{"referent_id": "ref_1", "textual_anchor": "he", "candidate_entity_ids": ["patient_1"]}]
    }
    state_2 = updater.update(state_1, input_t2)

    assert len(state_2.unresolved_referents) == 1
    assert state_2.unresolved_referents[0].status == ReferentStatus.RESOLVED
    assert state_2.unresolved_referents[0].resolved_entity_id == "patient_1"


def test_state_expiration():
    """Test state expiration removing stale entities beyond max turn freshness."""
    updater = SBDSUpdater(max_turn_freshness=2)
    state_0 = SharedBidirectionalDialogueState(sequence_id="seq_test_3")

    # Turn 1: Add entity
    state_1 = updater.update(state_0, {"entities": [{"entity_id": "temp_ent", "name": "Temp", "entity_type": "OBJECT"}]})
    assert len(state_1.active_entities) == 1

    # Turn 2: No mention of temp_ent
    state_2 = updater.update(state_1, {"english_text": "Irrelevant turn 2"})
    assert len(state_2.active_entities) == 1

    # Turn 3: No mention of temp_ent
    state_3 = updater.update(state_2, {"english_text": "Irrelevant turn 3"})
    assert len(state_3.active_entities) == 1

    # Turn 4: Turn freshness (4 - 1 = 3 > max_turn_freshness 2) -> Expired
    state_4 = updater.update(state_3, {"english_text": "Irrelevant turn 4"})
    assert len(state_4.active_entities) == 0


def test_confidence_decay():
    """Test exponential confidence decay on inactive entities across turns."""
    updater = SBDSUpdater(confidence_decay_factor=0.90)
    state_0 = SharedBidirectionalDialogueState(sequence_id="seq_test_4")

    # Turn 1: Entity confidence 1.0 at turn 1
    state_1 = updater.update(state_0, {"entities": [{"entity_id": "e_decay", "name": "DecayItem", "entity_type": "OBJECT", "confidence": 1.0}]})
    assert state_1.active_entities[0].confidence == 1.0

    # Turn 2: Inactive for 1 turn -> 1.0 * (0.90^1) = 0.90
    state_2 = updater.update(state_1, {"english_text": "Next turn"})
    assert state_2.active_entities[0].confidence == 0.90


def test_contradiction_resolution():
    """Test resolving slot value contradictions with higher-confidence current input."""
    updater = SBDSUpdater()
    state_0 = SharedBidirectionalDialogueState(sequence_id="seq_test_5")

    # Turn 1: Intent slot location = "Room 101"
    state_1 = updater.update(state_0, {"intent": {"intent_type": "APPOINTMENT", "domain": "CLINIC", "slot_values": {"room": "Room 101"}}})
    assert state_1.dialogue_intent.slot_values["room"] == "Room 101"

    # Turn 2: Updated slot location = "Room 202" (Overwrites contradiction)
    state_2 = updater.update(state_1, {"intent": {"intent_type": "APPOINTMENT", "domain": "CLINIC", "slot_values": {"room": "Room 202"}}})
    assert state_2.dialogue_intent.slot_values["room"] == "Room 202"


def test_deterministic_multi_turn_dialogue():
    """Test deterministic multi-turn dialogue simulation with SBDSReplayBuffer."""
    updater = SBDSUpdater()
    replay = SBDSReplayBuffer()
    current_state = SharedBidirectionalDialogueState(sequence_id="multi_turn_seq")
    replay.append(current_state)

    turns_data = [
        {"speaker_id": "PATIENT", "english_text": "Hello Doctor", "entities": [{"entity_id": "p1", "name": "Patient", "entity_type": "PERSON"}]},
        {"speaker_id": "DOCTOR", "english_text": "What is your symptom?", "intent": {"intent_type": "DIAGNOSIS"}},
        {"speaker_id": "PATIENT", "english_text": "Fever since yesterday", "temporal": {"time_frame": "PAST", "tense": "PAST"}},
        {"speaker_id": "DOCTOR", "english_text": "Take this medicine", "entities": [{"entity_id": "med1", "name": "Paracetamol", "entity_type": "OBJECT"}]}
    ]

    for turn_info in turns_data:
        current_state = updater.update(current_state, turn_info)
        replay.append(current_state)

    trajectory = replay.get_trajectory()
    assert len(trajectory) == 5  # Initial state + 4 turns
    assert trajectory[4].turn_metadata.turn_index == 4
    assert len(trajectory[4].active_entities) == 2
