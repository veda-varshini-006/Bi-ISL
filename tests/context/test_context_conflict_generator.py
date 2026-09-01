"""Unit tests for Context-Conflict Generator (Prompt 39)."""

import pytest
from src.context.sbds_schema import SharedBidirectionalDialogueState, Intent, Entity, EntityType
from src.context.context_conflict_generator import ContextConflictGenerator, CorruptionLevel


def test_context_conflict_generator_all_5_levels():
    """Test generating all 5 history corruption levels with provenance tracking."""
    gen = ContextConflictGenerator(seed=42)
    base_state = SharedBidirectionalDialogueState(
        sequence_id="seq_100",
        dialogue_intent=Intent(intent_type="CONSULTATION", domain="MEDICAL", slot_values={"room": "101"})
    )

    for level in CorruptionLevel:
        corr_state, provenance = gen.generate_corrupted_context(base_state, level)

        assert provenance["original_sequence_id"] == "seq_100"
        assert provenance["corruption_level"] == level.value
        assert len(provenance["modifications_applied"]) > 0

        if level == CorruptionLevel.CONTRADICTORY_HISTORY:
            assert "CONTRADICTION" in corr_state.dialogue_intent.slot_values["room"]
        elif level == CorruptionLevel.IRRELEVANT_HISTORY:
            assert corr_state.dialogue_intent.domain == "METEOROLOGY"
