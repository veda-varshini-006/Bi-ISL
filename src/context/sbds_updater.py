"""Shared Bidirectional Dialogue State (SBDS) Updater Subsystem (Prompt 32).

Updates SBDS objects deterministically across dialogue turns.

Input:
- previous_state: SharedBidirectionalDialogueState
- confirmed_input_output: Dict[str, Any] (text, glosses, entities, intent, etc.)
- confidence: float
- direction: CommunicationDirection

Output:
- new_state: SharedBidirectionalDialogueState (Immutable new version)

Handles:
- Entity creation & update
- Pronoun / reference tracking & resolution
- State expiration (max turn freshness threshold)
- Confidence decay over inactive turns
- Contradiction resolution
"""

from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import numpy as np

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
)


class SBDSUpdater:
    """Deterministic dialogue state updater for SBDS context management."""

    def __init__(
        self,
        max_turn_freshness: int = 5,
        confidence_decay_factor: float = 0.95,
        default_speaker: str = "USER_SIGNER"
    ):
        self.max_turn_freshness = max_turn_freshness
        self.confidence_decay_factor = confidence_decay_factor
        self.default_speaker = default_speaker

    def update(
        self,
        previous_state: SharedBidirectionalDialogueState,
        confirmed_input_output: Dict[str, Any],
        confidence: float = 1.0,
        direction: CommunicationDirection = CommunicationDirection.ISL_TO_ENGLISH
    ) -> SharedBidirectionalDialogueState:
        """Process turn update and return new immutable SBDS state version."""
        current_turn_idx = previous_state.turn_metadata.turn_index + 1
        speaker_id = confirmed_input_output.get("speaker_id", self.default_speaker)

        updated_entities: List[Entity] = []
        for ent in previous_state.active_entities:
            turns_inactive = current_turn_idx - ent.last_seen_turn
            if turns_inactive <= self.max_turn_freshness:
                decayed_conf = round(ent.confidence * (self.confidence_decay_factor ** turns_inactive), 4)
                updated_entities.append(
                    Entity(
                        entity_id=ent.entity_id,
                        name=ent.name,
                        entity_type=ent.entity_type,
                        confidence=decayed_conf,
                        attributes=ent.attributes,
                        last_seen_turn=ent.last_seen_turn
                    )
                )

        new_entities_input = confirmed_input_output.get("entities", [])
        entity_map = {e.entity_id: e for e in updated_entities}

        for raw_ent in new_entities_input:
            eid = raw_ent.get("entity_id", f"ent_{len(entity_map)+1}")
            ename = raw_ent.get("name", "UNNAMED")
            etype_str = raw_ent.get("entity_type", "CONCEPT")
            etype = EntityType(etype_str) if etype_str in EntityType.__members__ else EntityType.CONCEPT
            ent_conf = round(float(raw_ent.get("confidence", confidence)), 4)
            ent_attrs = raw_ent.get("attributes", {})

            if eid in entity_map:
                existing = entity_map[eid]
                merged_attrs = deepcopy(existing.attributes)
                merged_attrs.update(ent_attrs)
                entity_map[eid] = Entity(
                    entity_id=eid,
                    name=ename if ename != "UNNAMED" else existing.name,
                    entity_type=etype,
                    confidence=max(existing.confidence, ent_conf),
                    attributes=merged_attrs,
                    last_seen_turn=current_turn_idx
                )
            else:
                entity_map[eid] = Entity(
                    entity_id=eid,
                    name=ename,
                    entity_type=etype,
                    confidence=ent_conf,
                    attributes=ent_attrs,
                    last_seen_turn=current_turn_idx
                )

        active_entities_list = list(entity_map.values())

        text = confirmed_input_output.get("english_text", "")
        raw_referents = confirmed_input_output.get("referents", [])
        updated_referents: List[Referent] = []

        for ref in previous_state.unresolved_referents:
            if ref.status == ReferentStatus.UNRESOLVED:
                matched_id = None
                for ent in active_entities_list:
                    if ent.last_seen_turn == current_turn_idx:
                        matched_id = ent.entity_id
                        break
                if matched_id:
                    updated_referents.append(
                        Referent(
                            referent_id=ref.referent_id,
                            textual_anchor=ref.textual_anchor,
                            candidate_entity_ids=ref.candidate_entity_ids,
                            resolved_entity_id=matched_id,
                            status=ReferentStatus.RESOLVED
                        )
                    )
                else:
                    updated_referents.append(ref)

        for rr in raw_referents:
            ref_id = rr.get("referent_id", f"ref_{len(updated_referents)+1}")
            anchor = rr.get("textual_anchor", "pronoun")
            candidates = rr.get("candidate_entity_ids", [e.entity_id for e in active_entities_list])
            resolved_id = rr.get("resolved_entity_id", None)
            if resolved_id is None and candidates:
                resolved_id = candidates[0]
                status = ReferentStatus.RESOLVED
            elif resolved_id:
                status = ReferentStatus.RESOLVED
            else:
                status = ReferentStatus.UNRESOLVED

            updated_referents.append(
                Referent(
                    referent_id=ref_id,
                    textual_anchor=anchor,
                    candidate_entity_ids=candidates,
                    resolved_entity_id=resolved_id,
                    status=status
                )
            )

        raw_intent = confirmed_input_output.get("intent", {})
        intent_type = raw_intent.get("intent_type", previous_state.dialogue_intent.intent_type)
        domain = raw_intent.get("domain", previous_state.dialogue_intent.domain)
        new_slots = raw_intent.get("slot_values", {})
        intent_conf = round(float(raw_intent.get("confidence", confidence)), 4)

        merged_slots = deepcopy(previous_state.dialogue_intent.slot_values)
        merged_slots.update(new_slots)

        new_intent = Intent(
            intent_type=intent_type,
            domain=domain,
            slot_values=merged_slots,
            confidence=intent_conf
        )

        raw_temp = confirmed_input_output.get("temporal", {})
        new_temp = TemporalContext(
            time_frame=raw_temp.get("time_frame", previous_state.temporal_attributes.time_frame),
            tense=raw_temp.get("tense", previous_state.temporal_attributes.tense),
            relative_offset_seconds=float(raw_temp.get("relative_offset_seconds", 0.0))
        )

        raw_loc = confirmed_input_output.get("location", {})
        new_loc = LocationContext(
            location_name=raw_loc.get("location_name", previous_state.location_attributes.location_name),
            room_type=raw_loc.get("room_type", previous_state.location_attributes.room_type)
        )

        new_turn_meta = TurnMetadata(
            turn_index=current_turn_idx,
            active_speaker=speaker_id,
            direction=direction,
            system_mode="CONTEXT_GATED"
        )

        new_translation = ConfirmedTranslation(
            turn_id=f"turn_{current_turn_idx}",
            speaker_id=speaker_id,
            isl_glosses=confirmed_input_output.get("isl_glosses", []),
            english_text=text,
            confidence=confidence
        )

        new_confidence_meta = StateConfidence(
            overall_confidence=round(confidence, 4),
            entity_confidence=round(float(np.mean([e.confidence for e in active_entities_list])) if active_entities_list else 1.0, 4),
            intent_confidence=intent_conf
        )

        return SharedBidirectionalDialogueState(
            state_version="v1.0.0",
            sequence_id=previous_state.sequence_id,
            turn_metadata=new_turn_meta,
            active_entities=active_entities_list,
            dialogue_intent=new_intent,
            unresolved_referents=updated_referents,
            temporal_attributes=new_temp,
            location_attributes=new_loc,
            confidence_metadata=new_confidence_meta,
            last_confirmed_translation=new_translation
        )
