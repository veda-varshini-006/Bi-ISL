"""UGSA Adaptation Decision Gate (Prompt 45).

Gating Equation:
    g_t = 1[ p_t >= tau_p AND q_t >= tau_q AND delta_L_safe <= epsilon ]

where:
- p_t: Calibrated confidence metric
- q_t: Sequence agreement / consensus rate
- delta_L_safe: Safety set degradation metric
- tau_p, tau_q, epsilon: Validation-derived thresholds.

Logs decision telemetry to JSONL file:
- accept_reason
- reject_reason
- p_t
- q_t
- safety_delta
- timestamp
- signer_id
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class UGSAGate:
    """Unsupervised Group/Signer Adaptation decision gate."""

    def __init__(
        self,
        tau_p: float = 0.85,
        tau_q: float = 0.75,
        epsilon: float = 0.05,
        log_file_path: str = "./artifacts/logs/ugsa_gate_decisions.jsonl"
    ):
        self.tau_p = tau_p
        self.tau_q = tau_q
        self.epsilon = epsilon
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        p_t: float,
        q_t: float,
        safety_delta: float,
        signer_id: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate UGSA adaptation decision gate g_t."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        rejection_reasons = []
        if p_t < self.tau_p:
            rejection_reasons.append(f"LOW_CONFIDENCE (p_t={p_t:.3f} < tau_p={self.tau_p:.3f})")
        if q_t < self.tau_q:
            rejection_reasons.append(f"LOW_CONSENSUS (q_t={q_t:.3f} < tau_q={self.tau_q:.3f})")
        if safety_delta > self.epsilon:
            rejection_reasons.append(f"SAFETY_DEGRADATION (delta={safety_delta:.3f} > eps={self.epsilon:.3f})")

        gate_decision = 1 if len(rejection_reasons) == 0 else 0
        accept_reason = "CONFIDENCE_CONSENSUS_AND_SAFETY_VERIFIED" if gate_decision == 1 else ""
        reject_reason = "; ".join(rejection_reasons) if gate_decision == 0 else ""

        record = {
            "timestamp": timestamp,
            "signer_id": signer_id,
            "gate_decision": gate_decision,
            "p_t": round(float(p_t), 4),
            "q_t": round(float(q_t), 4),
            "safety_delta": round(float(safety_delta), 4),
            "tau_p": self.tau_p,
            "tau_q": self.tau_q,
            "epsilon": self.epsilon,
            "accept_reason": accept_reason,
            "reject_reason": reject_reason
        }

        self._log_decision(record)
        return record

    def _log_decision(self, record: Dict[str, Any]) -> None:
        """Appends decision record to JSONL log."""
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
