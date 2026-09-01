"""Unit tests for ReverseISLDomainSpecification (Prompt 61)."""

import os
import tempfile
import pytest

from src.english_to_isl.domain_specification import ReverseISLDomainSpecification


def test_domain_specification_intents_and_entities():
    """Test intents and entities list completeness."""
    spec = ReverseISLDomainSpecification()

    assert len(spec.INTENTS) == 8
    assert "SYMPTOM_REPORT" in spec.INTENTS
    assert "LOCATION_DIRECTION" in spec.INTENTS

    assert len(spec.ENTITIES) == 12
    assert "Doctor" in spec.ENTITIES
    assert "Pharmacy" in spec.ENTITIES


def test_is_word_supported():
    """Test supported vocabulary checker."""
    spec = ReverseISLDomainSpecification()

    assert spec.is_word_supported("doctor") is True
    assert spec.is_word_supported("fever") is True
    assert spec.is_word_supported("chemotherapy") is False


def test_check_risk_boundary_pass_and_reject():
    """Test risk boundary auditing for safe vs unsafe queries."""
    spec = ReverseISLDomainSpecification()

    safe_res = spec.check_risk_boundary("I have a fever today.")
    assert safe_res["is_safe"] is True
    assert safe_res["status"] == "APPROVED"

    unsafe_res = spec.check_risk_boundary("Patient needs emergency surgery and oncology treatment.")
    assert unsafe_res["is_safe"] is False
    assert unsafe_res["status"] == "REJECTED_RISK_BOUNDARY"
    assert "surgery" in unsafe_res["violated_terms"]


def test_export_domain_specification():
    """Test exporting JSON and MD specification files."""
    spec = ReverseISLDomainSpecification()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, data = spec.export_domain_specification()

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert data["domain_name"] == "ROUTINE_HEALTHCARE_PUBLIC_SERVICE"


def test_documentation_file_exists():
    """Verify REVERSE_ENGLISH_TO_ISL_DOMAIN_SPEC.md exists."""
    doc_path = "./docs/domain/REVERSE_ENGLISH_TO_ISL_DOMAIN_SPEC.md"
    assert os.path.exists(doc_path)
