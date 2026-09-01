"""Clarification Engine Module (Prompt 56).

Implements minimal clarification workflow for non-forced translations:
- "Please repeat the sign." (Visual uncertainty / blur)
- "Did you mean X or Y?" (Beam decode ambiguity)
- "Translation confidence is low." (General low confidence)

SAFETY ENFORCEMENT:
- Strictly prohibits generating unsupported medical or legal advice during clarification.

USABILITY METRIC LOGGING:
- Records clarification_frequency (%) across translation sessions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ClarificationEngine:
    """Clarification workflow engine with medical/legal safety guardrails."""

    UNSUPPORTED_TERMS = [
        "diagnose", "prescribe", "legal liability", "guarantee cure", "treatment plan", "verdict"
    ]

    def __init__(self, log_dir: str = "./artifacts/logs/usability"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.total_queries = 0
        self.clarifications_triggered = 0

    def generate_clarification_prompt(
        self,
        abstention_info: Dict[str, Any],
        top_candidates: Optional[List[str]] = None,
        domain: str = "GENERAL"
    ) -> Dict[str, Any]:
        """Generates user-facing clarification message with domain safety checks."""
        self.total_queries += 1
        reasons = abstention_info.get("abstain_reasons", [])

        if "LOW_VISUAL_CONFIDENCE" in reasons or "UNKNOWN_SIGN_OOV" in reasons:
            message = "Please repeat the sign."
            clarification_type = "REPEAT_SIGN"
        elif top_candidates and len(top_candidates) >= 2:
            cand_a = self._sanitize_safety(top_candidates[0], domain)
            cand_b = self._sanitize_safety(top_candidates[1], domain)
            message = f"Did you mean '{cand_a}' or '{cand_b}'?"
            clarification_type = "DISAMBIGUATION"
        else:
            message = "Translation confidence is low. Please sign again clearly."
            clarification_type = "LOW_CONFIDENCE_FALLBACK"

        for term in self.UNSUPPORTED_TERMS:
            if term in message.lower():
                message = "Translation confidence is low. Please repeat the sign."

        self.clarifications_triggered += 1
        clarification_freq = round((self.clarifications_triggered / float(self.total_queries)) * 100.0, 2)

        result = {
            "clarification_message": message,
            "clarification_type": clarification_type,
            "domain": domain,
            "safety_passed": True,
            "clarification_frequency_pct": clarification_freq,
            "total_queries": self.total_queries,
            "clarifications_triggered": self.clarifications_triggered
        }

        log_file = self.log_dir / "clarification_usability.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def _sanitize_safety(self, text: str, domain: str) -> str:
        """Sanitizes text to prevent unsupported medical/legal advice."""
        if domain.upper() in ("MEDICAL", "LEGAL"):
            for term in self.UNSUPPORTED_TERMS:
                if term in text.lower():
                    return "[UNVERIFIED_TERM]"
        return text
