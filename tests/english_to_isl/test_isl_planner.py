"""Unit tests for ISLPlanner covering all 8 supported intents (Prompt 65)."""

import os
import pytest

from src.english_to_isl.semantic_parser import EnglishSemanticParser
from src.english_to_isl.isl_planner import ISLPlanner


def test_plan_all_8_supported_intents():
    """Test planning IR objects for all 8 supported intents."""
    parser = EnglishSemanticParser()
    planner = ISLPlanner()

    test_sentences = [
        ("I have a fever today.", "ont:intent/symptom_report"),
        ("Where is the pharmacy?", "ont:intent/location_inquiry"),
        ("Take medicine in the morning.", "ont:intent/medication_instruction"),
        ("I want to book an appointment with doctor.", "ont:intent/appointment_booking"),
        ("I need urgent help!", "ont:intent/emergency_request"),
        ("I want to register.", "ont:intent/registration_checkin"),
        ("Where do I pay hospital bill?", "ont:intent/payment_billing"),
        ("Please repeat that.", "ont:intent/general_clarification")
    ]

    for text, expected_intent in test_sentences:
        frame = parser.parse_text(text)
        ir = planner.plan_semantic_frame(frame)

        assert ir["version"] == "1.0.0"
        assert len(ir["ordered_gloss_ids"]) >= 1
        assert len(ir["spatial_references"]) == len(ir["ordered_gloss_ids"])
        assert "provenance" in ir


def test_isl_grammar_ordering():
    """Test ISL Topic-Comment / Time-SOV ordering."""
    parser = EnglishSemanticParser()
    planner = ISLPlanner()

    frame = parser.parse_text("I have a fever today.")
    ir = planner.plan_semantic_frame(frame)

    # Time -> Topic -> Action
    assert ir["ordered_gloss_ids"] == ["TODAY", "FEVER", "HAVE"]


def test_non_manual_markers_wh_questions():
    """Test attaching non-manual facial expression markers for location inquiry."""
    parser = EnglishSemanticParser()
    planner = ISLPlanner()

    frame = parser.parse_text("Where is the pharmacy?")
    ir = planner.plan_semantic_frame(frame)

    assert any(m["marker"] == "eyebrows_furrowed" for m in ir["non_manual_markers"])


def test_documentation_file_exists():
    """Verify ISL_PLANNER_SPEC.md exists."""
    doc_path = "./docs/planning/ISL_PLANNER_SPEC.md"
    assert os.path.exists(doc_path)
