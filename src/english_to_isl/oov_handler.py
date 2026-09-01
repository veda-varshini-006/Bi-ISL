"""Robust Out-of-Vocabulary (OOV) Handler Module (Prompt 67).

STRICT RULE: Never substitutes a random nearest sign.

Resolution Modes:
1. FINGERSPELLING_VALIDATED: Manual alphabet fingerspelling
2. CLARIFICATION_REQUEST: Non-forced clarification prompt
3. TEXT_DISPLAY_FALLBACK: Text overlay caption fallback
4. EXPLICIT_OOV_STATE: Explicit OOV state halt with zero hallucination

Logs telemetry events to artifacts/logs/oov/oov_events.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class OOVHandler:
    """Robust OOV handler preventing random nearest sign substitution."""

    VALIDATED_FINGERSPELLING_VOCAB = set("abcdefghijklmnopqrstuvwxyz")

    def __init__(self, log_dir: str = "./artifacts/logs/oov"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "oov_events.jsonl"

    def handle_oov_term(self, oov_word: str, preferred_mode: str = "FINGERSPELLING_VALIDATED") -> Dict[str, Any]:
        """Handles an unmapped OOV word using safe, deterministic resolution modes."""
        word_clean = oov_word.strip().lower()

        # NEVER ALLOW RANDOM NEAREST NEIGHBOR SUBSTITUTION
        is_random_substituted = False

        if preferred_mode == "FINGERSPELLING_VALIDATED":
            fs_glosses = [f"FS_{char.upper()}" for char in word_clean if char in self.VALIDATED_FINGERSPELLING_VOCAB]
            resolution_data = {
                "mode": "FINGERSPELLING_VALIDATED",
                "fingerspelled_glosses": fs_glosses,
                "description": f"Fingerspelled '{word_clean}' using ISL manual alphabet."
            }
        elif preferred_mode == "CLARIFICATION_REQUEST":
            resolution_data = {
                "mode": "CLARIFICATION_REQUEST",
                "clarification_prompt": f"The term '{word_clean}' is not recognized in current ISL domain. Could you please rephrase?",
                "description": "Triggered non-forced clarification dialogue request."
            }
        elif preferred_mode == "TEXT_DISPLAY_FALLBACK":
            resolution_data = {
                "mode": "TEXT_DISPLAY_FALLBACK",
                "text_caption": f"[Unsigned term: {word_clean}]",
                "description": "Exposed text caption overlay alongside avatar rendering."
            }
        else:
            resolution_data = {
                "mode": "EXPLICIT_OOV_STATE",
                "status": "OOV_HALT",
                "description": f"Halted avatar generation for unmapped term '{word_clean}' with zero hallucination."
            }

        event_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "oov_word": word_clean,
            "resolution_mode": resolution_data["mode"],
            "resolution_details": resolution_data,
            "is_random_nearest_substituted": is_random_substituted
        }

        self._log_event(event_record)

        return event_record

    def _log_event(self, record: Dict[str, Any]) -> None:
        """Appends OOV event to JSONL telemetry log."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
