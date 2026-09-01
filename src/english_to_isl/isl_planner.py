"""ISL Planner Module (Prompt 65).

Transforms JSON Semantic Frames into Schema-Validated ISL Intermediate Representation (IR):
Pipeline Phases:
1. Semantic Transformation
2. Word/Sign Ordering (Topic-Comment / SOV ISL Grammar)
3. Non-Manual Markers (Facial Expressions & Head Movements)
4. Avatar Motion Hints (Spatial Loci & Co-articulation Hints)

Covers all 8 supported domain intents.
"""

from typing import Dict, List, Optional, Tuple, Any

from src.english_to_isl.isl_ir_schema import ISLIntermediateRepresentation


class ISLPlanner:
    """Rule-based ISL planner converting semantic frames into ISL IR objects."""

    def __init__(self):
        self.ir_validator = ISLIntermediateRepresentation()

    def plan_semantic_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Executes 4-phase ISL planning on input semantic frame."""
        intent = frame.get("intent_uri", "ont:intent/general_inquiry")

        # Phase 1: Semantic Transformation
        raw_concepts = self._phase1_semantic_transformation(frame)

        # Phase 2: Word / Sign Ordering (ISL Grammar: Time-Topic-Comment / SOV)
        ordered_glosses = self._phase2_sign_ordering(intent, raw_concepts, frame)

        # Phase 3: Non-Manual Markers (Facial & Head annotations)
        non_manuals = self._phase3_non_manual_markers(intent, frame, ordered_glosses)

        # Phase 4: Avatar Motion Hints (Spatial loci & transition hints)
        spatial_refs, transition_hints, timing = self._phase4_avatar_motion_hints(ordered_glosses)

        ir_object = {
            "version": "1.0.0",
            "intent": intent,
            "ordered_gloss_ids": ordered_glosses,
            "spatial_references": spatial_refs,
            "non_manual_markers": non_manuals,
            "timing": timing,
            "transition_hints": transition_hints,
            "oov_markers": [],
            "confidence": frame.get("confidence_score", 0.9),
            "provenance": {
                "parser": "EnglishSemanticParser_v1",
                "planner": "ISLPlanner_v1",
                "timestamp": "2026-09-02T00:16:00Z"
            }
        }

        is_valid, errors = self.ir_validator.validate_ir(ir_object)
        if not is_valid:
            raise ValueError(f"Generated ISL IR failed schema validation: {errors}")

        return ir_object

    def _phase1_semantic_transformation(self, frame: Dict[str, Any]) -> List[str]:
        """Extracts core ISL concepts from entities and slots."""
        concepts = []
        for ent in frame.get("entities", []):
            surf = ent.get("surface_word", "").upper()
            if surf and surf not in concepts:
                concepts.append(surf)
        return concepts

    def _phase2_sign_ordering(self, intent: str, concepts: List[str], frame: Dict[str, Any]) -> List[str]:
        """Applies ISL grammatical ordering (Time -> Topic -> Comment / SOV)."""
        temp = frame.get("temporality_uri")

        glosses = []
        if temp == "ont:temp/present_now":
            glosses.append("TODAY")
        elif temp == "ont:temp/future_tomorrow":
            glosses.append("TOMORROW")
        elif temp == "ont:temp/tod_morning":
            glosses.append("MORNING")

        if intent == "ont:intent/symptom_report":
            for c in concepts:
                glosses.append(c)
            if "HAVE" not in glosses:
                glosses.append("HAVE")
        elif intent == "ont:intent/location_inquiry":
            for c in concepts:
                glosses.append(c)
            glosses.append("WHERE")
        elif intent == "ont:intent/medication_instruction":
            glosses.append("MEDICINE")
            glosses.append("TAKE")
        elif intent == "ont:intent/appointment_booking":
            glosses.append("DOCTOR")
            glosses.append("APPOINTMENT")
            glosses.append("BOOK")
            glosses.append("WANT")
        elif intent == "ont:intent/emergency_request":
            glosses.append("HELP")
            glosses.append("EMERGENCY")
            glosses.append("URGENT")
        elif intent == "ont:intent/registration_checkin":
            glosses.append("PATIENT")
            glosses.append("REGISTER")
            glosses.append("WANT")
        elif intent == "ont:intent/payment_billing":
            glosses.append("BILL")
            glosses.append("PAYMENT")
            glosses.append("WHERE")
        elif intent == "ont:intent/general_clarification":
            glosses.append("PLEASE")
            glosses.append("REPEAT")
        else:
            glosses.extend(concepts if concepts else ["HELP"])

        return glosses

    def _phase3_non_manual_markers(self, intent: str, frame: Dict[str, Any], glosses: List[str]) -> List[Dict[str, str]]:
        """Attaches non-manual facial markers based on question type / intent."""
        markers = []
        qtype = frame.get("question_type_uri")

        if qtype == "ont:qtype/location_where" or "WHERE" in glosses:
            markers.append({"gloss_id": "WHERE", "marker": "eyebrows_furrowed"})
        elif intent in ["ont:intent/symptom_report", "ont:intent/appointment_booking"]:
            markers.append({"gloss_id": glosses[0] if glosses else "DOCTOR", "marker": "head_nod_slight"})
        elif intent == "ont:intent/emergency_request":
            markers.append({"gloss_id": "URGENT", "marker": "eyes_wide_expressive"})

        return markers

    def _phase4_avatar_motion_hints(self, glosses: List[str]) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]:
        """Attaches spatial loci and co-articulation transition hints."""
        spatial_refs = []
        for i, g in enumerate(glosses):
            locus = "LOC_CENTER" if i % 2 == 0 else "LOC_RIGHT"
            spatial_refs.append({"gloss_id": g, "locus": locus})

        transition_hints = ["smooth_blend"]
        timing = {
            "duration_ms": max(600, len(glosses) * 500),
            "speed_multiplier": 1.0
        }

        return spatial_refs, transition_hints, timing
