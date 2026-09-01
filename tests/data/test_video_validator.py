"""Unit tests for Bi-ISL Video Validation & Quarantine Subsystem."""

import os
import tempfile
import pytest

from src.data.schema import CanonicalDataSample
from src.data.video_validator import VideoValidator


def test_zero_byte_video_detection_and_quarantine():
    """Test detecting 0-byte video files and moving them to quarantine."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        q_dir = os.path.join(tmp_dir, "quarantine")
        validator = VideoValidator(quarantine_dir=q_dir)

        # Create zero-byte dummy video file
        corrupt_video = os.path.join(tmp_dir, "corrupt_video.mp4")
        open(corrupt_video, "w").close()

        insp = validator.inspect_video(corrupt_video)
        assert insp["zero_byte"] is True
        assert "ZERO_BYTE_FILE" in insp["issues"]

        samples = [CanonicalDataSample(sample_id="v1", dataset="TEST", video_path=corrupt_video)]
        stats, results = validator.validate_dataset_videos(samples, quarantine_corrupt=True)

        assert stats["quarantined_videos_count"] == 1
        assert not os.path.exists(corrupt_video)  # Moved out of original path
        assert os.path.exists(os.path.join(q_dir, "corrupt_video.mp4"))  # Quarantined


def test_missing_file_detection():
    """Test detecting non-existent video file path."""
    validator = VideoValidator()
    insp = validator.inspect_video("/path/does/not/exist.mp4")
    assert "FILE_NOT_FOUND" in insp["issues"]


def test_dataset_health_report_export():
    """Test exporting JSON and Markdown video health reports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        validator = VideoValidator(quarantine_dir=tmp_dir)

        samples = [
            CanonicalDataSample(sample_id="v1", dataset="TEST", video_path=os.path.join(tmp_dir, "valid.mp4"))
        ]
        # Create small dummy file
        with open(samples[0].video_path, "wb") as f:
            f.write(b"dummy_video_bytes_content_for_health_report_test")

        stats, results = validator.validate_dataset_videos(samples, quarantine_corrupt=False)

        json_rep, md_rep = validator.export_health_reports(stats, results, output_dir=tmp_dir)
        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Bi-ISL Video Dataset Health Report" in md_text
        assert "Total Videos Audited:" in md_text
