"""Context-Conflict & Perturbation Stress-Test Generator (Prompt 39).

Generates 5 history corruption levels for E3 stress testing:
1. CORRECT_HISTORY: Unmodified ground-truth SBDS dialogue history.
2. IRRELEVANT_HISTORY: Unrelated dialogue context (e.g. sports/weather vs medical).
3. SEMANTICALLY_RELATED_WRONG: Related domain with swapped entity values.
4. PARTIALLY_MISLEADING: Partial slot value noise.
5. CONTRADICTORY_HISTORY: Explicitly conflicting slot values (e.g., Room 101 vs Room 202, Past vs Future).

STRICT RULES:
- Preserves current visual input.
- Stores provenance metadata for every modified context.
- NEVER alters test labels (ground-truth target text remains untouched).
"""

from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

from src.context.sbds_schema import (
    SharedBidirectionalDialogueState,
    Entity,
    EntityType,
    Intent,
    Referent,
    TemporalContext,
    LocationContext,
)


class CorruptionLevel(str, Enum):
    CORRECT_HISTORY = "CORRECT_HISTORY"
    IRRELEVANT_HISTORY = "IRRELEVANT_HISTORY"
    SEMANTICALLY_RELATED_WRONG = "SEMANTICALLY_RELATED_WRONG"
    PARTIALLY_MISLEADING = "PARTIALLY_MISLEADING"
    CONTRADICTORY_HISTORY = "CONTRADICTORY_HISTORY"


class ContextConflictGenerator:
    """Stress-test context generator producing history corruptions with provenance tracking."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_corrupted_context(
        self,
        base_state: SharedBidirectionalDialogueState,
        corruption_level: CorruptionLevel
    ) -> Tuple[SharedBidirectionalDialogueState, Dict[str, Any]]:
        """Generate corrupted SBDS state and provenance dictionary."""
        provenance = {
            "original_sequence_id": base_state.sequence_id,
            "original_state_hash": base_state.compute_state_hash(),
            "corruption_level": corruption_level.value,
            "modifications_applied": []
        }

        if corruption_level == CorruptionLevel.CORRECT_HISTORY:
            provenance["modifications_applied"].append("None (Ground-truth history)")
            return base_state, provenance

        corrupted_dict = base_state.model_dump()

        if corruption_level == CorruptionLevel.IRRELEVANT_HISTORY:
            corrupted_dict["dialogue_intent"] = {
                "intent_type": "WEATHER_QUERY",
                "domain": "METEOROLOGY",
                "slot_values": {"condition": "rainy"},
                "confidence": 0.9
            }
            corrupted_dict["active_entities"] = []
            corrupted_dict["location_attributes"] = {"location_name": "Stadium", "room_type": "OUTDOORS"}
            provenance["modifications_applied"].append("Replaced domain with METEOROLOGY and cleared entities.")

        elif corruption_level == CorruptionLevel.SEMANTICALLY_RELATED_WRONG:
            corrupted_dict["active_entities"] = [
                {"entity_id": "e_swapped", "name": "Dr. Kapoor", "entity_type": "PERSON", "confidence": 0.85, "last_seen_turn": 1}
            ]
            corrupted_dict["dialogue_intent"]["slot_values"] = {"specialty": "Cardiology", "department": "ICU"}
            provenance["modifications_applied"].append("Swapped entity Dr. Sharma -> Dr. Kapoor and specialty -> Cardiology.")

        elif corruption_level == CorruptionLevel.PARTIALLY_MISLEADING:
            slots = deepcopy(base_state.dialogue_intent.slot_values)
            slots["noisy_slot"] = "misleading_value"
            corrupted_dict["dialogue_intent"]["slot_values"] = slots
            provenance["modifications_applied"].append("Injected misleading slot key.")

        elif corruption_level == CorruptionLevel.CONTRADICTORY_HISTORY:
            corrupted_dict["dialogue_intent"] = {
                "intent_type": base_state.dialogue_intent.intent_type,
                "domain": base_state.dialogue_intent.domain,
                "slot_values": {"room": "Room 999_CONTRADICTION", "status": "CANCELLED"},
                "confidence": 0.99
            }
            corrupted_dict["temporal_attributes"] = {"time_frame": "FUTURE", "tense": "FUTURE", "relative_offset_seconds": 3600.0}
            provenance["modifications_applied"].append("Injected explicit room/temporal contradictions.")

        new_state = SharedBidirectionalDialogueState.from_dict(corrupted_dict)
        provenance["corrupted_state_hash"] = new_state.compute_state_hash()

        return new_state, provenance
