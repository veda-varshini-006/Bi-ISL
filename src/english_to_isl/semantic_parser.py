"""English Semantic Parser Module (Prompt 63).

Parses English text into structured JSON Semantic Frames:
- intent_uri (ont:intent/...)
- entities (List of extracted entity URIs & slots)
- question_type_uri (ont:qtype/...)
- negation_type_uri (ont:negation/...)
- temporality_uri (ont:temp/...)
- confidence_score (float in [0.0, 1.0])
- parse_status (SUCCESS, UNCERTAIN, REJECTED_OUT_OF_SCOPE)

STRICT RULE: Never maps directly from arbitrary English surface words to avatar clips.
All translations pass through structured semantic frames.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.english_to_isl.domain_specification import ReverseISLDomainSpecification
from src.english_to_isl.domain_ontology import DomainOntology


class EnglishSemanticParser:
    """Deterministic / template-assisted semantic parser converting English text to JSON frames."""

    def __init__(self):
        self.domain_spec = ReverseISLDomainSpecification()
        self.ontology = DomainOntology()

    def parse_text(self, english_text: str) -> Dict[str, Any]:
        """Parses English input text into structured JSON semantic frame."""
        text_clean = english_text.strip()
        lowered = text_clean.lower()

        risk_audit = self.domain_spec.check_risk_boundary(text_clean)
        if not risk_audit["is_safe"]:
            return {
                "english_text": english_text,
                "intent_uri": "ont:intent/out_of_scope",
                "entities": [],
                "confidence_score": 0.0,
                "parse_status": "REJECTED_OUT_OF_SCOPE",
                "rejection_reason": f"Violated risk boundary terms: {risk_audit['violated_terms']}",
                "direct_clip_mapping_allowed": False
            }

        intent_uri = "ont:intent/general_inquiry"
        if any(w in lowered for w in ["fever", "cough", "pain", "headache", "hurt", "bleed"]):
            intent_uri = "ont:intent/symptom_report"
        elif any(w in lowered for w in ["where", "location", "direction", "find", "way"]):
            intent_uri = "ont:intent/location_inquiry"
        elif any(w in lowered for w in ["take", "medicine", "pill", "dose", "drug"]):
            intent_uri = "ont:intent/medication_instruction"
        elif any(w in lowered for w in ["appointment", "book", "schedule", "meet"]):
            intent_uri = "ont:intent/appointment_booking"
        elif any(w in lowered for w in ["help", "emergency", "ambulance", "urgent"]):
            intent_uri = "ont:intent/emergency_request"

        entities = []
        if "doctor" in lowered:
            entities.append({"entity_uri": "ont:entity/medical_practitioner", "slot": "target_entity", "surface_word": "doctor"})
        if "pharmacy" in lowered:
            entities.append({"entity_uri": "ont:loc/pharmacy", "slot": "location_target", "surface_word": "pharmacy"})
        if "fever" in lowered:
            entities.append({"entity_uri": "ont:entity/symptom_fever", "slot": "symptom_target", "surface_word": "fever"})
        if "cough" in lowered:
            entities.append({"entity_uri": "ont:entity/symptom_cough", "slot": "symptom_target", "surface_word": "cough"})
        if "medicine" in lowered or "pill" in lowered:
            entities.append({"entity_uri": "ont:entity/medication", "slot": "target_entity", "surface_word": "medicine"})

        qtype_uri = None
        if "where" in lowered:
            qtype_uri = "ont:qtype/location_where"
        elif "when" in lowered:
            qtype_uri = "ont:qtype/time_when"
        elif "who" in lowered:
            qtype_uri = "ont:qtype/identity_who"
        elif text_clean.endswith("?"):
            qtype_uri = "ont:qtype/binary_yesno"

        negation_uri = None
        if any(w in lowered.split() for w in ["no", "not", "never", "none", "don't"]):
            negation_uri = "ont:negation/absent"

        temp_uri = None
        if "morning" in lowered:
            temp_uri = "ont:temp/tod_morning"
        elif "night" in lowered:
            temp_uri = "ont:temp/tod_night"
        elif "today" in lowered:
            temp_uri = "ont:temp/present_now"
        elif "tomorrow" in lowered:
            temp_uri = "ont:temp/future_tomorrow"

        known_words = sum(1 for w in lowered.split() if self.domain_spec.is_word_supported(w))
        total_words = max(1, len(lowered.split()))
        confidence_score = round(min(1.0, (known_words / float(total_words)) + 0.1), 3)

        parse_status = "SUCCESS" if confidence_score >= 0.4 else "UNCERTAIN"

        return {
            "english_text": english_text,
            "intent_uri": intent_uri,
            "entities": entities,
            "question_type_uri": qtype_uri,
            "negation_type_uri": negation_uri,
            "temporality_uri": temp_uri,
            "confidence_score": confidence_score,
            "parse_status": parse_status,
            "direct_clip_mapping_allowed": False
        }
