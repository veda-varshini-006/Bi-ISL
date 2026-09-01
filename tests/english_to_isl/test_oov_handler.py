"""Unit tests for OOVHandler (Prompt 67)."""

import os
import tempfile
import pytest

from src.english_to_isl.oov_handler import OOVHandler


def test_oov_fingerspelling_mode():
    """Test fingerspelling resolution mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = OOVHandler(log_dir=tmp_dir)

        res = handler.handle_oov_term("paracetamol", preferred_mode="FINGERSPELLING_VALIDATED")

        assert res["resolution_mode"] == "FINGERSPELLING_VALIDATED"
        assert res["is_random_nearest_substituted"] is False
        assert "FS_P" in res["resolution_details"]["fingerspelled_glosses"]


def test_oov_clarification_mode():
    """Test clarification request resolution mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = OOVHandler(log_dir=tmp_dir)

        res = handler.handle_oov_term("chemotherapy", preferred_mode="CLARIFICATION_REQUEST")

        assert res["resolution_mode"] == "CLARIFICATION_REQUEST"
        assert res["is_random_nearest_substituted"] is False
        assert "chemotherapy" in res["resolution_details"]["clarification_prompt"]


def test_oov_text_display_mode():
    """Test text display overlay resolution mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = OOVHandler(log_dir=tmp_dir)

        res = handler.handle_oov_term("oncology", preferred_mode="TEXT_DISPLAY_FALLBACK")

        assert res["resolution_mode"] == "TEXT_DISPLAY_FALLBACK"
        assert res["is_random_nearest_substituted"] is False
        assert "[Unsigned term: oncology]" in res["resolution_details"]["text_caption"]


def test_oov_explicit_halt_mode():
    """Test explicit OOV state halt mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = OOVHandler(log_dir=tmp_dir)

        res = handler.handle_oov_term("biopsy", preferred_mode="EXPLICIT_OOV_STATE")

        assert res["resolution_mode"] == "EXPLICIT_OOV_STATE"
        assert res["is_random_nearest_substituted"] is False
        assert res["resolution_details"]["status"] == "OOV_HALT"


def test_oov_telemetry_logging():
    """Test appending event record to JSONL log file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = OOVHandler(log_dir=tmp_dir)
        handler.handle_oov_term("ibuprofen")

        assert os.path.exists(handler.log_file)
        with open(handler.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert "ibuprofen" in lines[0]


def test_documentation_file_exists():
    """Verify OOV_HANDLING_SPEC.md exists."""
    doc_path = "./docs/handling/OOV_HANDLING_SPEC.md"
    assert os.path.exists(doc_path)
