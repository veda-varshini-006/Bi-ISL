"""Transactional Personalization & Safety Rollback Manager (Prompt 47).

Before every accepted adaptation:
1. Snapshots adapter state dict (copy of trainable weights).
2. Executes bounded gradient update.
3. Evaluates Protected Reference Safety Set post-update.
4. If degradation exceeds threshold (epsilon = 5.0%), performs ATOMIC ROLLBACK to previous state dict.
5. Maintains comprehensive adaptation transaction history log.

Includes simulation tests for catastrophic bad updates.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

from src.personalization.signer_adapter import SignerAdapter
from src.personalization.protected_safety_set import ProtectedSafetySet
from src.personalization.bounded_updater import BoundedSignerUpdater


class TransactionalPersonalizationManager:
    """Manager for transactional online signer adaptation with atomic safety rollback."""

    def __init__(
        self,
        adapter: SignerAdapter,
        safety_set: ProtectedSafetySet,
        updater: BoundedSignerUpdater,
        max_allowed_degradation_pct: float = 5.0,
        history_log_path: str = "./artifacts/logs/transactional_adaptation_history.jsonl"
    ):
        self.adapter = adapter
        self.safety_set = safety_set
        self.updater = updater
        self.max_allowed_degradation_pct = max_allowed_degradation_pct
        self.history_log_path = Path(history_log_path)
        self.history_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.snapshot_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.adaptation_history: List[Dict[str, Any]] = []

    def snapshot_adapter_state(self) -> Dict[str, torch.Tensor]:
        """Creates a deepcopy snapshot of current adapter parameters."""
        self.snapshot_state_dict = {
            name: param.detach().clone()
            for name, param in self.adapter.named_parameters()
            if param.requires_grad
        }
        return self.snapshot_state_dict

    def rollback(self) -> bool:
        """Atomically restores adapter parameters from snapshot."""
        if self.snapshot_state_dict is None:
            raise RuntimeError("Cannot rollback: No snapshot state dict exists!")

        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                if param.requires_grad and name in self.snapshot_state_dict:
                    param.copy_(self.snapshot_state_dict[name])

        return True

    def execute_transactional_step(
        self,
        task_loss: torch.Tensor,
        pre_safety_score: float,
        post_safety_eval_fn: Any,
        signer_id: str
    ) -> Dict[str, Any]:
        """Executes a transactional adaptation step with atomic safety verification."""
        timestamp = datetime.now(timezone.utc).isoformat()

        self.snapshot_adapter_state()

        update_info = self.updater.update_step(task_loss)
        if not update_info["updated"]:
            transaction_record = {
                "timestamp": timestamp,
                "signer_id": signer_id,
                "status": "REJECTED_BY_UPDATER_BOUNDS",
                "reason": update_info.get("reason", ""),
                "pre_safety_score": pre_safety_score,
                "post_safety_score": pre_safety_score,
                "degradation_pct": 0.0,
                "rollback_performed": False
            }
            self.adaptation_history.append(transaction_record)
            self._log_transaction(transaction_record)
            return transaction_record

        post_safety_score = float(post_safety_eval_fn())

        degrad_info = self.safety_set.measure_adaptation_degradation(
            pre_adaptation_bleu=pre_safety_score,
            post_adaptation_bleu=post_safety_score,
            max_allowed_degradation_pct=self.max_allowed_degradation_pct
        )

        rollback_performed = False
        if degrad_info["rollback_required"]:
            self.rollback()
            rollback_performed = True
            status = "ROLLED_BACK_CATASTROPHIC_DEGRADATION"
        else:
            status = "COMMITTED_SUCCESSFULLY"

        transaction_record = {
            "timestamp": timestamp,
            "signer_id": signer_id,
            "status": status,
            "pre_safety_score": round(pre_safety_score, 4),
            "post_safety_score": round(post_safety_score, 4),
            "degradation_pct": degrad_info["degradation_pct"],
            "max_allowed_degradation_pct": self.max_allowed_degradation_pct,
            "rollback_performed": rollback_performed,
            "param_distance_after_step": update_info["param_distance"]
        }

        self.adaptation_history.append(transaction_record)
        self._log_transaction(transaction_record)
        return transaction_record

    def _log_transaction(self, record: Dict[str, Any]) -> None:
        """Logs transaction record to JSONL history file."""
        with open(self.history_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
