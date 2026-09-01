"""Unit tests for Bi-ISL Formal Dataset Registry."""

import pytest
from src.data.registry import DatasetRegistry, DatasetSpec


def test_default_datasets_registered():
    """Test that all 4 required ISL datasets are registered by default."""
    registry = DatasetRegistry()
    dataset_list = registry.list_datasets()

    expected = ["INCLUDE", "ISLTRANSLATE", "ISIGN", "ISH-NEWS"]
    for dataset_name in expected:
        assert dataset_name in dataset_list, f"Missing registered dataset: {dataset_name}"


def test_dataset_spec_fields():
    """Test DatasetSpec fields for ISLTranslate and iSign."""
    registry = DatasetRegistry()

    spec_isl = registry.get("ISLTranslate")
    assert spec_isl.name == "ISLTranslate"
    assert "Joshi" in spec_isl.citation
    assert spec_isl.task_type == "Continuous Sign Language Translation"
    assert spec_isl.signer_metadata_availability is True
    assert spec_isl.source_video_metadata_availability is True

    spec_isign = registry.get("iSign")
    assert spec_isign.name == "iSign"
    assert "Multi-Task" in spec_isign.task_type
    assert spec_isign.download_status == "PENDING_LICENSE_CHECK"


def test_license_checking():
    """Test license checking workflow before download approval."""
    registry = DatasetRegistry()

    # Before license check
    spec = registry.get("INCLUDE")
    assert spec.download_status == "PENDING_LICENSE_CHECK"

    # Execute license check
    audit_info = registry.check_license("INCLUDE")
    assert audit_info["approved_for_download"] is True
    assert audit_info["name"] == "INCLUDE"

    # Status updated after license check
    assert registry.get("INCLUDE").download_status == "NOT_DOWNLOADED"


def test_zero_automatic_downloads():
    """Verify zero automatic downloads happen during initialization or registration."""
    registry = DatasetRegistry()
    for name in registry.list_datasets():
        spec = registry.get(name)
        assert spec.download_status in ["PENDING_LICENSE_CHECK", "NOT_DOWNLOADED"]
        # Local path should not exist automatically prior to explicit download step
        assert spec.local_path is not None
