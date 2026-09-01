"""Context Reliability Signals Estimator (Prompt 34).

Estimates multi-dimensional context reliability features for SBDS states:
1. visual_confidence
2. context_age
3. entity_overlap
4. intent_compatibility
5. semantic_similarity
6. contradiction_indicators
7. num_unresolved_referents
8. context_confidence
9. turn_distance

Logs all signals for explainability analysis and exports reliability_signals.jsonl.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict

from src.context.sbds_schema import SharedBidirectionalDialogueState, ReferentStatus


class ReliabilitySignals(BaseModel):
    """Structured Pydantic schema for context reliability signals."""
    
    model_config = ConfigDict(frozen=True)

    sequence_id: str
    turn_index: int
    visual_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    context_age_seconds: float = Field(ge=0.0, default=0.0)
    entity_overlap_score: float = Field(ge=0.0, le=1.0, default=1.0)
    intent_compatibility: float = Field(ge=0.0, le=1.0, default=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0, default=1.0)
    contradiction_indicator: float = Field(ge=0.0, le=1.0, default=0.0)
    num_unresolved_referents: int = Field(ge=0, default=0)
    context_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    turn_distance: int = Field(ge=0, default=0)
    overall_reliability_score: float = Field(ge=0.0, le=1.0, default=1.0)


class ContextReliabilityEstimator:
    """Estimates context reliability signals for SBDS state objects."""

    def __init__(self, log_dir: str = "./artifacts/logs/context"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def estimate(
        self,
        state: SharedBidirectionalDialogueState,
        current_observation: Dict[str, Any],
        visual_confidence: float = 1.0,
        current_turn: int = 1
    ) -> ReliabilitySignals:
        """Estimate 9 candidate reliability signals and compute composite reliability score."""
        seq_id = state.sequence_id
        state_turn = state.turn_metadata.turn_index
        turn_dist = max(0, current_turn - state_turn)
        ctx_age = float(turn_dist * 3.0)

        state_eids = {e.entity_id for e in state.active_entities}
        obs_eids = {e.get("entity_id") for e in current_observation.get("entities", []) if e.get("entity_id")}
        if state_eids and obs_eids:
            entity_overlap = len(state_eids.intersection(obs_eids)) / len(state_eids.union(obs_eids))
        else:
            entity_overlap = 1.0 if not state_eids and not obs_eids else 0.5

        obs_intent = current_observation.get("intent", {}).get("intent_type", "UNKNOWN")
        if state.dialogue_intent.intent_type == "UNKNOWN" or obs_intent == "UNKNOWN":
            intent_compat = 0.8
        elif state.dialogue_intent.intent_type == obs_intent:
            intent_compat = 1.0
        else:
            intent_compat = 0.3

        prev_text = state.last_confirmed_translation.english_text if state.last_confirmed_translation else ""
        curr_text = current_observation.get("english_text", "")
        prev_words = set(prev_text.split())
        curr_words = set(curr_text.split())
        if prev_words and curr_words:
            sem_sim = len(prev_words.intersection(curr_words)) / len(prev_words.union(curr_words))
        else:
            sem_sim = 0.5

        contradiction = 0.0
        obs_slots = current_observation.get("intent", {}).get("slot_values", {})
        for k, v in obs_slots.items():
            if k in state.dialogue_intent.slot_values and state.dialogue_intent.slot_values[k] != v:
                contradiction = 0.8
                break

        num_unresolved = sum(1 for r in state.unresolved_referents if r.status == ReferentStatus.UNRESOLVED)
        ctx_conf = state.confidence_metadata.overall_confidence

        penalty_turn = min(0.5, turn_dist * 0.1)
        penalty_referents = min(0.3, num_unresolved * 0.1)

        raw_score = (
            0.25 * visual_confidence +
            0.20 * entity_overlap +
            0.20 * intent_compat +
            0.20 * ctx_conf +
            0.15 * (1.0 - contradiction)
        ) - (penalty_turn + penalty_referents)

        composite_score = round(max(0.0, min(1.0, raw_score)), 4)

        signals = ReliabilitySignals(
            sequence_id=seq_id,
            turn_index=current_turn,
            visual_confidence=round(visual_confidence, 4),
            context_age_seconds=round(ctx_age, 2),
            entity_overlap_score=round(entity_overlap, 4),
            intent_compatibility=round(intent_compat, 4),
            semantic_similarity=round(sem_sim, 4),
            contradiction_indicator=round(contradiction, 4),
            num_unresolved_referents=num_unresolved,
            context_confidence=round(ctx_conf, 4),
            turn_distance=turn_dist,
            overall_reliability_score=composite_score
        )

        self.log_signals(signals)
        return signals

    def log_signals(self, signals: ReliabilitySignals) -> str:
        """Log signals to reliability_signals.jsonl for explainability analysis."""
        logfile = self.log_dir / "reliability_signals.jsonl"
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(signals.model_dump()) + "\n")
        return str(logfile)
