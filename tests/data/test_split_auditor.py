"""Unit tests for Bi-ISL Split Auditor and Experiment E0 Leakage Audit."""

import os
import tempfile
import pytest

from src.data.schema import CanonicalDataSample
from src.data.split_auditor import SplitAuditor, LeakageAuditFailedError
from src.experiments.e0_leakage_audit import E0LeakageAuditExperiment


def test_clean_splits_pass_audit():
    """Test that clean, disjoint splits pass audit without errors."""
    auditor = SplitAuditor()
    splits = {
        "train": [
            CanonicalDataSample(sample_id="tr_1", dataset="INCLUDE", source_video_id="v1", signer_id="s1", text="hello"),
            CanonicalDataSample(sample_id="tr_2", dataset="INCLUDE", source_video_id="v2", signer_id="s2", text="world")
        ],
        "test": [
            CanonicalDataSample(sample_id="te_1", dataset="INCLUDE", source_video_id="v3", signer_id="s3", text="goodbye")
        ]
    }

    summary = auditor.audit_splits(splits)
    assert summary["total_violations"] == 0
    assert summary["critical_violations"] == 0
    assert summary["source_video_leakage_detected"] is False


def test_source_video_leakage_fails_loudly():
    """Test that source video leakage across train and test fails loudly."""
    auditor = SplitAuditor()
    leaky_splits = {
        "train": [
            CanonicalDataSample(sample_id="tr_1", dataset="INCLUDE", source_video_id="v_shared", signer_id="s1", text="hello")
        ],
        "test": [
            CanonicalDataSample(sample_id="te_1", dataset="INCLUDE", source_video_id="v_shared", signer_id="s2", text="hello")
        ]
    }

    with pytest.raises(LeakageAuditFailedError, match="Confirmed source-video leakage detected"):
        auditor.audit_splits(leaky_splits)


def test_signer_leakage_detected():
    """Test detecting signer overlap across signer-disjoint splits."""
    auditor = SplitAuditor()
    signer_leaky_splits = {
        "train": [
            CanonicalDataSample(sample_id="tr_1", dataset="INCLUDE", source_video_id="v1", signer_id="s_overlapping", text="a")
        ],
        "test": [
            CanonicalDataSample(sample_id="te_1", dataset="INCLUDE", source_video_id="v2", signer_id="s_overlapping", text="b")
        ]
    }

    # If source video is different, source_video_leakage_detected is False, but signer leakage is flagged
    summary = auditor.audit_splits(signer_leaky_splits, signer_disjoint_required=True)
    assert summary["total_violations"] == 1
    assert summary["violations"][0]["type"] == "SIGNER_LEAKAGE"


def test_temporal_segment_overlap_detected():
    """Test detecting overlapping temporal segments on the same source video."""
    auditor = SplitAuditor()
    overlap_splits = {
        "train": [
            CanonicalDataSample(
                sample_id="tr_1", dataset="ISLTranslate", source_video_id="v_long",
                segment_start=0.0, segment_end=5.0, signer_id="s1", text="part one"
            )
        ],
        "dev": [
            CanonicalDataSample(
                sample_id="dev_1", dataset="ISLTranslate", source_video_id="v_long",
                segment_start=3.0, segment_end=8.0, signer_id="s2", text="part two"
            )
        ]
    }

    with pytest.raises(LeakageAuditFailedError):
        auditor.audit_splits(overlap_splits)


def test_export_json_csv_markdown_reports():
    """Test exporting JSON, CSV, and Markdown audit reports."""
    auditor = SplitAuditor()
    splits = {
        "train": [CanonicalDataSample(sample_id="tr_1", dataset="INCLUDE", source_video_id="v1", signer_id="s1", text="a")],
        "test": [CanonicalDataSample(sample_id="te_1", dataset="INCLUDE", source_video_id="v2", signer_id="s2", text="b")]
    }

    summary = auditor.audit_splits(splits)
    with tempfile.TemporaryDirectory() as tmp_dir:
        jpath, cpath, mpath = auditor.export_reports(summary, output_dir=tmp_dir)

        assert os.path.exists(jpath)
        assert os.path.exists(cpath)
        assert os.path.exists(mpath)

        with open(mpath, "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "Experiment E0: Dataset Leakage Audit Report" in md_content
        assert "Zero data leakage" in md_content


def test_e0_experiment_runner():
    """Test running E0LeakageAuditExperiment end-to-end."""
    exp = E0LeakageAuditExperiment()
    summary = exp.run()
    assert summary["total_violations"] == 0
