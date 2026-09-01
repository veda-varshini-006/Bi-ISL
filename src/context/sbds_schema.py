"""Shared Bidirectional Dialogue State (SBDS) Schema (Prompt 31).

Defines typed, state-versioned, immutable SBDS representations for Bi-ISL context-gated sign language translation.

Represents:
- active_entities: Structured entity instances with confidence and attributes
- dialogue_intent: Active dialogue intent, domain, and slot values
- unresolved_referents: Pronoun and spatial gesture referent resolution targets
- temporal_attributes: Time frame, relative offsets, and grammatical tense context
- location_attributes: Spatial location, coordinates, and room/enclosure metadata
- confidence_metadata: Multimodal state confidence scores across modalities
- last_confirmed_translation: Last validated ISL-to-English / English-to-ISL turn translation
- turn_metadata: Turn index, active speaker, and communication direction

Supports:
- State versioning and SHA-256 state hashing
- JSON/Dictionary serialization and deserialization
- State replay buffer for reproducible dialogue trajectory execution

STRICT RULE: Does NOT store arbitrary full transcript as the primary state.
"""

from datetime import datetime, timezone
import hashlib
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict


class CommunicationDirection(str, Enum):
    ISL_TO_ENGLISH = "ISL_TO_ENGLISH"
    ENGLISH_TO_ISL = "ENGLISH_TO_ISL"
    BIDIRECTIONAL = "BIDIRECTIONAL"


class EntityType(str, Enum):
    PERSON = "PERSON"
    OBJECT = "OBJECT"
    LOCATION = "LOCATION"
    ACTION = "ACTION"
    CONCEPT = "CONCEPT"


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    entity_id: str
    name: str
    entity_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    last_seen_turn: int = 0


class Intent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    intent_type: str = "UNKNOWN"
    domain: str = "GENERAL"
    slot_values: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ReferentStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


class Referent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    referent_id: str
    textual_anchor: str
    temporal_anchor: str = ""
    candidate_entity_ids: List[str] = Field(default_factory=list)
    resolved_entity_id: Optional[str] = None
    status: ReferentStatus = ReferentStatus.UNRESOLVED


class TemporalContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    time_frame: str = "PRESENT"
    absolute_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relative_offset_seconds: float = 0.0
    tense: str = "PRESENT"


class LocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    location_name: str = "UNKNOWN"
    coordinates: Optional[List[float]] = None
    spatial_relation: str = "NEAR"
    room_type: str = "GENERAL"


class StateConfidence(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    overall_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    entity_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    intent_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    temporal_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    location_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ConfirmedTranslation(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    turn_id: str
    speaker_id: str
    isl_glosses: List[str] = Field(default_factory=list)
    english_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TurnMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    turn_index: int = 0
    active_speaker: str = "USER_SIGNER"
    direction: CommunicationDirection = CommunicationDirection.ISL_TO_ENGLISH
    system_mode: str = "CONTEXT_GATED"


class SharedBidirectionalDialogueState(BaseModel):
    """Immutable, typed, versioned Shared Bidirectional Dialogue State (SBDS)."""
    
    model_config = ConfigDict(frozen=True)
    
    state_version: str = "v1.0.0"
    sequence_id: str = "default_seq_001"
    turn_metadata: TurnMetadata = Field(default_factory=TurnMetadata)
    active_entities: List[Entity] = Field(default_factory=list)
    dialogue_intent: Intent = Field(default_factory=Intent)
    unresolved_referents: List[Referent] = Field(default_factory=list)
    temporal_attributes: TemporalContext = Field(default_factory=TemporalContext)
    location_attributes: LocationContext = Field(default_factory=LocationContext)
    confidence_metadata: StateConfidence = Field(default_factory=StateConfidence)
    last_confirmed_translation: Optional[ConfirmedTranslation] = None

    def compute_state_hash(self) -> str:
        """Compute deterministic SHA-256 hash of structured state payload."""
        data_str = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to python dictionary."""
        d = self.model_dump()
        d["state_hash"] = self.compute_state_hash()
        return d

    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedBidirectionalDialogueState":
        """Deserialize state from dictionary."""
        data_copy = data.copy()
        data_copy.pop("state_hash", None)
        return cls(**data_copy)

    @classmethod
    def from_json(cls, json_str: str) -> "SharedBidirectionalDialogueState":
        """Deserialize state from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def create_next_version(self, **updates: Any) -> "SharedBidirectionalDialogueState":
        """Create new immutable state version with updated fields."""
        current_data = self.model_dump()
        current_data.update(updates)
        if "turn_metadata" in updates and isinstance(updates["turn_metadata"], dict):
            current_data["turn_metadata"] = updates["turn_metadata"]
        return SharedBidirectionalDialogueState(**current_data)


class SBDSReplayBuffer:
    """Replay buffer storing trajectory of state transitions for reproducible evaluation."""

    def __init__(self):
        self.trajectory: List[SharedBidirectionalDialogueState] = []

    def append(self, state: SharedBidirectionalDialogueState) -> None:
        self.trajectory.append(state)

    def get_trajectory(self) -> List[SharedBidirectionalDialogueState]:
        return self.trajectory

    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self.trajectory], indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SBDSReplayBuffer":
        buf = cls()
        data_list = json.loads(json_str)
        for item in data_list:
            buf.append(SharedBidirectionalDialogueState.from_dict(item))
        return buf
