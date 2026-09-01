"""Naive Template / Lookup English -> Sign Sequence Baseline Module (Prompt 68).

Implements an intentionally simple but valid baseline for E9 comparison:
- Direct surface word lookup & static string templates
- Preserves English SVO word order (no ISL SOV/Topic-Comment grammar rules)
- Omits non-manual markers (no facial expressions or head movements)
- Omits structured semantic frame abstraction (no ont:* URIs)
- Omits spatial loci & co-articulation hints
"""

from typing import Dict, List, Optional, Any


class NaiveReverseBaseline:
    """Naive lookup baseline mapping English text directly to gloss sequences."""

    SURFACE_LOOKUP_DICT = {
        "doctor": "DOCTOR",
        "hospital": "HOSPITAL",
        "fever": "FEVER",
        "cough": "COUGH",
        "pain": "PAIN",
        "headache": "HEADACHE",
        "medicine": "MEDICINE",
        "pill": "PILL",
        "pharmacy": "PHARMACY",
        "today": "TODAY",
        "tomorrow": "TOMORROW",
        "yesterday": "YESTERDAY",
        "morning": "MORNING",
        "night": "NIGHT",
        "where": "WHERE",
        "when": "WHEN",
        "help": "HELP",
        "register": "REGISTER",
        "appointment": "APPOINTMENT",
        "i": "ME",
        "have": "HAVE",
        "take": "TAKE",
        "book": "BOOK",
        "want": "WANT",
        "is": "IS",
        "the": "THE",
        "in": "IN"
    }

    def __init__(self):
        pass

    def translate_text(self, english_text: str) -> Dict[str, Any]:
        """Translates English surface text to sign sequence using naive surface lookup."""
        words = english_text.strip().lower().replace("?", "").replace(".", "").split()

        gloss_sequence = []
        for word in words:
            if word in self.SURFACE_LOOKUP_DICT:
                gloss_sequence.append(self.SURFACE_LOOKUP_DICT[word])
            else:
                gloss_sequence.append(word.upper())

        return {
            "english_text": english_text,
            "ordered_gloss_ids": gloss_sequence,
            "has_non_manual_markers": False,
            "is_structured_semantic": False,
            "preserves_isl_grammar": False,
            "baseline_name": "NaiveLookupBaseline"
        }
