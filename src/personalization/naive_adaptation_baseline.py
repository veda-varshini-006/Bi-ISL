"""Naive Signer Fine-Tuning Controlled Baseline (Prompt 48).

Controlled baseline for evaluating UGSA safety mechanisms:
- Uses EXACTLY the same adaptation samples, batch size, and learning rate as UGSA.
- NO confidence gating (updates are applied blindly to all input samples).
- NO protected-set safety rollback (degradation is ignored; updates commit permanently).

Purpose:
Determines whether UGSA confidence gating and transactional rollback mechanisms
are required to prevent catastrophic forgetting and parameter drift under noisy online adaptation.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim

from src.personalization.signer_adapter import SignerAdapter


class NaiveSignerFineTuningBaseline(nn.Module):
    """Ungated, unmonitored naive fine-tuning baseline."""

    def __init__(
        self,
        adapter: SignerAdapter,
        lr: float = 1e-4
    ):
        super().__init__()
        self.adapter = adapter
        self.lr = lr
        self.optimizer = optim.AdamW(
            [p for p in self.adapter.parameters() if p.requires_grad],
            lr=self.lr
        )
        self.step_count = 0

    def adapt_uncontrolled_step(self, task_loss: torch.Tensor) -> Dict[str, Any]:
        """Executes a single uncontrolled gradient update step without gate or rollback."""
        self.optimizer.zero_grad()
        task_loss.backward()
        self.optimizer.step()

        self.step_count += 1

        return {
            "step_count": self.step_count,
            "task_loss": float(task_loss.item()),
            "confidence_gate_applied": False,
            "safety_rollback_applied": False,
            "status": "UNCONTROLLED_COMMIT"
        }
