"""Unit tests for EnglishSemanticParser (Prompt 63)."""

import os
import pytest

from src.english_to_isl.semantic_parser import EnglishSemanticParser


def test_parse_text_symptom_report():
    """Test parsing English symptom report into JSON semantic frame."""
    parser = EnglishSemanticParser()

    frame = parser.parse_text("I have a fever today.")

    assert frame["intent_uri"] == "ont:intent/symptom_report"
    assert frame["parse_status"] == "SUCCESS"
    assert frame["confidence_score"] > 0.5
    assert len(frame["entities"]) >= 1
    assert frame["entities"][0]["entity_uri"] == "ont:entity/symptom_fever"


def test_parse_text_location_inquiry():
    """Test parsing English location inquiry with question type URI."""
    parser = EnglishSemanticParser()

    frame = parser.parse_text("Where is the pharmacy?")

    assert frame["intent_uri"] == "ont:intent/location_inquiry"
    assert frame["question_type_uri"] == "ont:qtype/location_where"
    assert frame["parse_status"] == "SUCCESS"


def test_parse_text_risk_boundary_rejection():
    """Test out-of-scope risk boundary rejection."""
    parser = EnglishSemanticParser()

    frame = parser.parse_text("Patient needs emergency surgery and oncology treatment.")

    assert frame["parse_status"] == "REJECTED_OUT_OF_SCOPE"
    assert frame["confidence_score"] == 0.0
    assert "surgery" in frame["rejection_reason"]


def test_direct_clip_mapping_disallowed():
    """Verify direct surface word to avatar clip mapping is strictly disallowed."""
    parser = EnglishSemanticParser()

    frame = parser.parse_text("Take medicine in the morning.")

    assert frame["direct_clip_mapping_allowed"] is False


def test_documentation_file_exists():
    """Verify SEMANTIC_PARSER_ARCHITECTURE.md exists."""
    doc_path = "./docs/parsing/SEMANTIC_PARSER_ARCHITECTURE.md"
    assert os.path.exists(doc_path)
