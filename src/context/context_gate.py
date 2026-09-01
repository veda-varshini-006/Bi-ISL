"""Context-Evidence Reliability Gate Subsystem (Prompt 35).

Implements the context-evidence reliability gate:
alpha_t = sigmoid(W_g [h_t ; c_t ; u_t] + b_g)
h_tilde = h_t + alpha_t * W_c(c_t)

Features:
- Inspectable alpha_t logged per example
- Support for ablation modes: LEARNED, FORCE_ZERO (alpha=0), FORCE_ONE (alpha=1), FIXED_CONSTANT
"""

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn


class GateAblationMode(str, Enum):
    LEARNED = "LEARNED"
    FORCE_ZERO = "FORCE_ZERO"
    FORCE_ONE = "FORCE_ONE"
    FIXED_CONSTANT = "FIXED_CONSTANT"


class ContextEvidenceGate(nn.Module):
    """Context-evidence reliability gate controlling context integration into visual features."""

    def __init__(
        self,
        embed_dim: int = 256,
        reliability_dim: int = 9,
        ablation_mode: GateAblationMode = GateAblationMode.LEARNED,
        fixed_alpha_value: float = 0.5,
        log_dir: str = "./artifacts/logs/context"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.reliability_dim = reliability_dim
        self.ablation_mode = ablation_mode
        self.fixed_alpha_value = fixed_alpha_value
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gate_proj = nn.Sequential(
            nn.Linear(2 * embed_dim + reliability_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.context_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        u_t: torch.Tensor,
        sample_ids: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass computing gated representation h_tilde and inspectable alpha_t."""
        if h_t.dim() == 3:
            h_pooled = h_t.mean(dim=1)
        else:
            h_pooled = h_t

        if c_t.dim() == 3:
            c_pooled = c_t.mean(dim=1)
        else:
            c_pooled = c_t

        batch_size = h_t.shape[0]

        if self.ablation_mode == GateAblationMode.LEARNED:
            concat_inputs = torch.cat([h_pooled, c_pooled, u_t], dim=-1)
            alpha_logits = self.gate_proj(concat_inputs)
            alpha_t = torch.sigmoid(alpha_logits)
        elif self.ablation_mode == GateAblationMode.FORCE_ZERO:
            alpha_t = torch.zeros(batch_size, 1, device=h_t.device)
        elif self.ablation_mode == GateAblationMode.FORCE_ONE:
            alpha_t = torch.ones(batch_size, 1, device=h_t.device)
        elif self.ablation_mode == GateAblationMode.FIXED_CONSTANT:
            alpha_t = torch.full((batch_size, 1), self.fixed_alpha_value, device=h_t.device)

        proj_context = self.context_proj(c_t)

        if h_t.dim() == 3:
            alpha_broadcast = alpha_t.unsqueeze(1)
        else:
            alpha_broadcast = alpha_t

        h_tilde = h_t + alpha_broadcast * proj_context

        alpha_list = alpha_t.squeeze(-1).tolist()
        if not isinstance(alpha_list, list):
            alpha_list = [alpha_list]

        diagnostics = {
            "ablation_mode": self.ablation_mode.value,
            "alpha_mean": float(alpha_t.mean().item()),
            "alpha_min": float(alpha_t.min().item()),
            "alpha_max": float(alpha_t.max().item()),
            "alpha_per_sample": alpha_list
        }

        if sample_ids and len(sample_ids) == batch_size:
            self.log_alpha_values(sample_ids, alpha_list)

        return h_tilde, alpha_t, diagnostics

    def log_alpha_values(self, sample_ids: List[str], alpha_values: List[float]) -> str:
        """Log per-example alpha_t values for explainability auditing."""
        logfile = self.log_dir / "gate_alpha_logs.jsonl"
        with open(logfile, "a", encoding="utf-8") as f:
            for sid, alpha in zip(sample_ids, alpha_values):
                f.write(json.dumps({"sample_id": sid, "alpha_t": round(alpha, 6), "mode": self.ablation_mode.value}) + "\n")
        return str(logfile)
