"""Unit tests for Bi-ISL Dataset Acquisition Adapters."""

import json
import os
import tempfile
import pytest

from src.data.adapter import (
    INCLUDEAdapter,
    ISLTranslateAdapter,
    iSignAdapter,
    ISHNewsAdapter
)
from src.data.registry import DatasetRegistry


def test_adapter_initialization_and_checksum():
    """Test adapter initialization, file hashing, and corruption detection."""
    registry = DatasetRegistry()
    registry.check_license("INCLUDE")
    adapter = INCLUDEAdapter(registry=registry)

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("test_content_for_checksum_audit")
        temp_path = f.name

    try:
        checksum = adapter.calculate_checksum(temp_path)
        assert len(checksum) == 64  # SHA-256 hex string length
        assert adapter.detect_corrupt_or_partial(temp_path, expected_checksum=checksum) is False
        assert adapter.detect_corrupt_or_partial(temp_path, expected_checksum="invalid_hash") is True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_license_permission_gating():
    """Test that download is blocked prior to license approval."""
    registry = DatasetRegistry()
    # Prior to license check, download_status is PENDING_LICENSE_CHECK
    adapter = INCLUDEAdapter(registry=registry)
    assert adapter.check_legal_permission() is False

    # Approve license check
    registry.check_license("INCLUDE")
    assert adapter.check_legal_permission() is True


def test_manual_download_instruction_generation():
    """Test that ISLTranslate generates manual download instructions."""
    registry = DatasetRegistry()
    registry.check_license("ISLTranslate")
    adapter = ISLTranslateAdapter(registry=registry)

    instructions = adapter.generate_manual_download_instructions()
    assert "MANUAL DATASET ACQUISITION INSTRUCTIONS FOR ISLTRANSLATE" in instructions
    assert "https://github.com/cfilt/ISLTranslate" in instructions
    assert adapter.acquire() is False


def test_manifest_writing_and_acquisition():
    """Test dataset acquisition workflow and dataset_manifest.json generation."""
    registry = DatasetRegistry()
    registry.check_license("INCLUDE")
    adapter = INCLUDEAdapter(registry=registry)

    success = adapter.acquire()
    assert success is True

    manifest_path = adapter.target_dir / "dataset_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_name"] == "INCLUDE"
    assert data["sample_count"] == 2
    assert len(data["samples"]) == 2
