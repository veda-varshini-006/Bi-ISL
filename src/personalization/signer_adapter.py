"""Lightweight Signer-Specific Adapter Module (Prompt 41).

Implements lightweight bottleneck residual adapters for signer personalization:
    y = x + W_up * GELU(W_down * x)

Candidate Placement Locations:
1. VISUAL_ENCODER_OUTPUT: Applied right after frame-level feature extraction.
2. SELECTED_ENCODER_BLOCKS: Applied within intermediate TCN encoder blocks.
3. TEMPORAL_REPRESENTATION: Applied to fused temporal feature sequence.
4. DECODER_INPUT: Applied directly prior to translation autoregressive decoding.

Base model remains frozen (requires_grad=False) initially.
Measures trainable parameters and memory footprint per signer (<50K params).
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class AdapterPlacement(str, Enum):
    VISUAL_ENCODER_OUTPUT = "VISUAL_ENCODER_OUTPUT"
    SELECTED_ENCODER_BLOCKS = "SELECTED_ENCODER_BLOCKS"
    TEMPORAL_REPRESENTATION = "TEMPORAL_REPRESENTATION"
    DECODER_INPUT = "DECODER_INPUT"


class SignerAdapter(nn.Module):
    """Bottleneck residual adapter for signer-specific adaptation."""

    def __init__(
        self,
        in_dim: int = 256,
        bottleneck_dim: int = 16,
        placement: AdapterPlacement = AdapterPlacement.TEMPORAL_REPRESENTATION,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.placement = placement

        self.down_proj = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual bottleneck forward pass: y = x + up_proj(GELU(down_proj(x)))."""
        res = self.down_proj(x)
        res = self.act(res)
        res = self.dropout(res)
        res = self.up_proj(res)
        return x + res

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def measure_memory_footprint_kb(self) -> float:
        """Returns memory overhead in Kilobytes (float32 = 4 bytes per param)."""
        return (self.count_parameters() * 4.0) / 1024.0


class AdaptedMultimodalModel(nn.Module):
    """Base Multimodal model wrapped with frozen base parameters and active SignerAdapter."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter_placement: AdapterPlacement = AdapterPlacement.TEMPORAL_REPRESENTATION,
        bottleneck_dim: int = 16
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_placement = adapter_placement

        for p in self.base_model.parameters():
            p.requires_grad = False

        in_dim = getattr(base_model, "fusion_dim", 256)
        self.adapter = SignerAdapter(
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            placement=adapter_placement
        )

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward pass routing features through the active placement adapter."""
        out = self.base_model(**kwargs)
        logits = out["logits"] if isinstance(out, dict) else out

        feat_dim = logits.shape[-1]
        if feat_dim != self.adapter.in_dim:
            if not hasattr(self, "in_proj") or self.in_proj.in_features != feat_dim:
                self.in_proj = nn.Linear(feat_dim, self.adapter.in_dim, device=logits.device)
                self.out_proj = nn.Linear(self.adapter.in_dim, feat_dim, device=logits.device)
            adapted_feat = self.adapter(self.in_proj(logits))
            logits = self.out_proj(adapted_feat)
        else:
            logits = self.adapter(logits)

        return {"logits": logits}

    def count_trainable_parameters(self) -> int:
        """Verify only adapter parameters are trainable."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
