"""Explainability & Decision Tracer Diagnostics Subsystem (Prompt 58).

Exposes research diagnostics detailing why a prediction changed per example:
- visual_confidence (p_t)
- context_gate_alpha (alpha_t)
- context_features
- ugsa_state
- adaptation_accepted (bool) & reject_reason
- rollback_event (bool)
- top_alternatives (List[str])

Exports to JSONL telemetry logs for research analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class DecisionTracer:
    """Diagnostic tracer capturing per-example inference telemetry."""

    def __init__(self, log_dir: str = "./artifacts/logs/explainability"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def trace_decision(
        self,
        sample_id: str,
        visual_confidence: float,
        context_gate_alpha: float,
        context_features: Optional[List[float]] = None,
        ugsa_state: str = "STABLE",
        adaptation_accepted: bool = True,
        reject_reason: Optional[str] = None,
        rollback_event: bool = False,
        top_alternatives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Traces and records diagnostic metadata per example."""
        record = {
            "sample_id": sample_id,
            "visual_confidence": round(visual_confidence, 4),
            "context_gate_alpha": round(context_gate_alpha, 4),
            "has_context_features": context_features is not None,
            "ugsa_state": ugsa_state,
            "adaptation_accepted": adaptation_accepted,
            "reject_reason": reject_reason,
            "rollback_event": rollback_event,
            "top_alternatives": top_alternatives or []
        }

        log_file = self.log_dir / "decision_telemetry.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record
