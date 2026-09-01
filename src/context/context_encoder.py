"""SBDS Context Encoder Subsystem (Prompt 33).

Converts Shared Bidirectional Dialogue State (SBDS) into compact fixed/sequence
representations usable by the translation decoder.

Architectures Supported:
1. SIMPLE_EMBEDDING: Projection layer over concatenated component embeddings.
2. TRANSFORMER_ATTENTION: Small 2-layer Transformer encoder with multi-head self-attention.

Features:
- Explicit boolean masks for missing state components.
- Zero dependency on large generative LLMs.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn

from src.context.sbds_schema import SharedBidirectionalDialogueState


class EncoderArchitecture(str, Enum):
    SIMPLE_EMBEDDING = "SIMPLE_EMBEDDING"
    TRANSFORMER_ATTENTION = "TRANSFORMER_ATTENTION"


class SBDSContextEncoder(nn.Module):
    """Compact SBDS context encoder transforming structured state into translation embeddings."""

    def __init__(
        self,
        embed_dim: int = 256,
        architecture: EncoderArchitecture = EncoderArchitecture.TRANSFORMER_ATTENTION,
        num_layers: int = 2,
        num_heads: int = 4,
        max_entities: int = 5,
        max_referents: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.architecture = architecture
        self.max_entities = max_entities
        self.max_referents = max_referents

        self.entity_proj = nn.Sequential(
            nn.Linear(5 + 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.intent_proj = nn.Sequential(
            nn.Linear(10, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.temporal_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.location_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.confidence_proj = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.num_components = 4 + max_entities + max_referents

        if self.architecture == EncoderArchitecture.SIMPLE_EMBEDDING:
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.num_components * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
        elif self.architecture == EncoderArchitecture.TRANSFORMER_ATTENTION:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def extract_state_features(
        self,
        states: List[SharedBidirectionalDialogueState],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert list of SBDS objects into (B, N, D) feature tensors and boolean mask."""
        batch_size = len(states)
        features = torch.zeros(batch_size, self.num_components, self.embed_dim, device=device)
        mask = torch.ones(batch_size, self.num_components, dtype=torch.bool, device=device)

        for b, state in enumerate(states):
            slot_idx = 0

            has_intent = state.dialogue_intent.intent_type != "UNKNOWN"
            if has_intent:
                raw_intent = torch.zeros(10, device=device)
                raw_intent[0] = state.dialogue_intent.confidence
                features[b, slot_idx] = self.intent_proj(raw_intent)
                mask[b, slot_idx] = True
            else:
                mask[b, slot_idx] = False
            slot_idx += 1

            has_temp = bool(state.temporal_attributes.time_frame)
            if has_temp:
                raw_temp = torch.tensor([1.0, float(state.temporal_attributes.relative_offset_seconds), 0.0, 0.0], device=device)
                features[b, slot_idx] = self.temporal_proj(raw_temp)
                mask[b, slot_idx] = True
            else:
                mask[b, slot_idx] = False
            slot_idx += 1

            has_loc = state.location_attributes.location_name != "UNKNOWN"
            if has_loc:
                raw_loc = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                features[b, slot_idx] = self.location_proj(raw_loc)
                mask[b, slot_idx] = True
            else:
                mask[b, slot_idx] = False
            slot_idx += 1

            cm = state.confidence_metadata
            raw_conf = torch.tensor([cm.overall_confidence, cm.entity_confidence, cm.intent_confidence, cm.temporal_confidence, cm.location_confidence], device=device)
            features[b, slot_idx] = self.confidence_proj(raw_conf)
            mask[b, slot_idx] = True
            slot_idx += 1

            for i in range(self.max_entities):
                if i < len(state.active_entities):
                    ent = state.active_entities[i]
                    raw_ent = torch.zeros(7, device=device)
                    raw_ent[0] = ent.confidence
                    raw_ent[1] = float(ent.last_seen_turn)
                    features[b, slot_idx] = self.entity_proj(raw_ent)
                    mask[b, slot_idx] = True
                else:
                    mask[b, slot_idx] = False
                slot_idx += 1

            for i in range(self.max_referents):
                if i < len(state.unresolved_referents):
                    features[b, slot_idx] = self.entity_proj(torch.zeros(7, device=device))
                    mask[b, slot_idx] = True
                else:
                    mask[b, slot_idx] = False
                slot_idx += 1

        return features, mask

    def forward(
        self,
        states: List[SharedBidirectionalDialogueState],
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass outputting context representations and missing component masks."""
        feats, mask = self.extract_state_features(states, device)

        if self.architecture == EncoderArchitecture.SIMPLE_EMBEDDING:
            B, N, D = feats.shape
            flat_feats = feats.reshape(B, N * D)
            context_output = self.fusion_layer(flat_feats)
        elif self.architecture == EncoderArchitecture.TRANSFORMER_ATTENTION:
            padding_mask = ~mask
            context_output = self.transformer_encoder(feats, src_key_padding_mask=padding_mask)

        return context_output, mask
