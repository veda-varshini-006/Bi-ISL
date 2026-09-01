"""Unit tests verifying Bi-ISL Baseline Reproduction Audit Document (Prompt 29)."""

import os
import pytest


def test_baseline_reproduction_document_exists():
    """Test BASELINE_REPRODUCTION.md file exists in docs/baselines/."""
    doc_path = "./docs/baselines/BASELINE_REPRODUCTION.md"
    assert os.path.exists(doc_path)


def test_baseline_reproduction_covers_all_required_fields():
    """Test BASELINE_REPRODUCTION.md covers all 8 required audit fields."""
    doc_path = "./docs/baselines/BASELINE_REPRODUCTION.md"
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Published Number" in content
    assert "Our Baseline Number" in content
    assert "Difference" in content
    assert "Split Differences" in content
    assert "Preprocessing Differences" in content
    assert "Architecture Differences" in content
    assert "Random Seed Variance" in content
    assert "Possible Explanations" in content
    assert "Disclaimer" in content or "Strict Non-Claim" in content
    assert "INCLUDE" in content
    assert "ISLTranslate" in content
    assert "iSign" in content
    assert "ISH-NEWS" in content
