"""Unit tests for ISLIntermediateRepresentation (Prompt 64)."""

import os
import tempfile
import pytest

from src.english_to_isl.isl_ir_schema import ISLIntermediateRepresentation


def test_validate_ir_valid_example():
    """Test validating valid canonical ISL IR example."""
    ir_manager = ISLIntermediateRepresentation()
    example = ir_manager.create_example_ir()

    is_valid, errors = ir_manager.validate_ir(example)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_ir_missing_field():
    """Test detection of missing required field in IR dict."""
    ir_manager = ISLIntermediateRepresentation()
    example = ir_manager.create_example_ir()
    del example["provenance"]

    is_valid, errors = ir_manager.validate_ir(example)
    assert is_valid is False
    assert any("provenance" in err for err in errors)


def test_validate_ir_invalid_version():
    """Test detection of invalid schema version string."""
    ir_manager = ISLIntermediateRepresentation()
    example = ir_manager.create_example_ir()
    example["version"] = "2.0.0"

    is_valid, errors = ir_manager.validate_ir(example)
    assert is_valid is False
    assert any("version" in err for err in errors)


def test_export_ir_spec():
    """Test exporting JSON and MD IR specification files."""
    ir_manager = ISLIntermediateRepresentation()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, data = ir_manager.export_ir_spec()

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)
        assert "json_schema" in data


def test_documentation_file_exists():
    """Verify ISL_INTERMEDIATE_REPRESENTATION_SPEC.md exists."""
    doc_path = "./docs/schemas/ISL_INTERMEDIATE_REPRESENTATION_SPEC.md"
    assert os.path.exists(doc_path)
