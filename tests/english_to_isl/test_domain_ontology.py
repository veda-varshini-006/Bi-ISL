"""Unit tests for DomainOntology (Prompt 62)."""

import os
import tempfile
import pytest

from src.english_to_isl.domain_ontology import DomainOntology


def test_resolve_concept_stable_uris():
    """Test resolution of stable concept URIs across 9 ontological dimensions."""
    ontology = DomainOntology()

    assert ontology.resolve_concept("ont:intent/symptom_report") is not None
    assert ontology.resolve_concept("ont:entity/doctor") is None  # Check correct key format
    assert ontology.resolve_concept("ont:entity/medical_practitioner") is not None
    assert ontology.resolve_concept("ont:slot/target_entity") is not None
    assert ontology.resolve_concept("ont:qtype/location_where") is not None
    assert ontology.resolve_concept("ont:negation/absent") is not None
    assert ontology.resolve_concept("ont:temp/tod_morning") is not None
    assert ontology.resolve_concept("ont:loc/pharmacy") is not None
    assert ontology.resolve_concept("ont:action/take_medication") is not None


def test_ontology_relationships():
    """Test entity-relationship triples in domain ontology graph."""
    ontology = DomainOntology()

    assert len(ontology.RELATIONSHIPS) >= 3
    rel1 = ontology.RELATIONSHIPS[0]
    assert rel1["subject"] == "ont:entity/patient"
    assert rel1["relation"] == "ont:rel/experiences"
    assert rel1["object"] == "ont:entity/symptom_fever"


def test_export_ontology():
    """Test exporting ontology specification files."""
    ontology = DomainOntology()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, data = ontology.export_ontology()

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert data["ontology_uri_prefix"] == "ont:"
        assert "intents" in data


def test_documentation_file_exists():
    """Verify DOMAIN_ONTOLOGY_SPEC.md exists."""
    doc_path = "./docs/ontology/DOMAIN_ONTOLOGY_SPEC.md"
    assert os.path.exists(doc_path)
