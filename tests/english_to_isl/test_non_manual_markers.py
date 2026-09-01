"""Unit tests for NonManualMarkerTaxonomy (Prompt 66)."""

import os
import tempfile
import pytest

from src.english_to_isl.non_manual_markers import NonManualMarkerTaxonomy


def test_non_manual_marker_categories():
    """Test presence of 7 core non-manual marker categories."""
    taxonomy = NonManualMarkerTaxonomy()

    categories = list(taxonomy.MARKER_TAXONOMY.keys())
    assert len(categories) == 7
    assert "eyebrow_movement" in categories
    assert "head_movement" in categories
    assert "facial_expression" in categories
    assert "mouth_pattern" in categories
    assert "body_lean" in categories
    assert "question_markers" in categories
    assert "negation" in categories


def test_get_marker_info_with_citation():
    """Test retrieving description and linguistic citation source for eyebrows_furrowed."""
    taxonomy = NonManualMarkerTaxonomy()

    info = taxonomy.get_marker_info("eyebrow_movement", "eyebrows_furrowed")
    assert info is not None
    assert "WH-questions" in info["description"]
    assert "Zeshan (2004)" in info["citation"]


def test_export_taxonomy_spec():
    """Test exporting JSON and MD taxonomy specification files."""
    taxonomy = NonManualMarkerTaxonomy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, data = taxonomy.export_taxonomy_spec()

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert data["categories_count"] == 7


def test_documentation_file_exists():
    """Verify NON_MANUAL_MARKERS_SPEC.md exists."""
    doc_path = "./docs/linguistics/NON_MANUAL_MARKERS_SPEC.md"
    assert os.path.exists(doc_path)
