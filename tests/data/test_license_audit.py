"""Unit tests verifying dataset licensing audit documentation compliance."""

import os
import pytest


def test_license_audit_document_exists():
    """Verify that docs/datasets/LICENSE_AUDIT.md exists and is non-empty."""
    audit_path = os.path.join("docs", "datasets", "LICENSE_AUDIT.md")
    assert os.path.exists(audit_path), f"Missing license audit document at {audit_path}"
    assert os.path.getsize(audit_path) > 500, "License audit document appears incomplete."


def test_license_audit_covers_all_datasets():
    """Verify that every intended dataset is documented with license terms."""
    audit_path = os.path.join("docs", "datasets", "LICENSE_AUDIT.md")
    with open(audit_path, "r", encoding="utf-8") as f:
        content = f.read()

    datasets = ["INCLUDE", "ISLTranslate", "iSign", "ISH-NEWS"]
    for dataset in datasets:
        assert dataset in content, f"Dataset '{dataset}' not documented in LICENSE_AUDIT.md"

    required_sections = [
        "Official Source",
        "Citation",
        "License",
        "Registration Requirements",
        "Redistribution Restrictions",
        "Allowed Research Use",
        "Preprocessing Outputs Sharing",
        "Checkpoint Distribution"
    ]
    for section in required_sections:
        assert section in content, f"Missing required license audit field: '{section}'"
