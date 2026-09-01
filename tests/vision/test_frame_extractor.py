"""Unit tests for Bi-ISL Frame Extraction and Preprocessing Pipeline."""

import pytest
import numpy as np

from src.vision.frame_extractor import FrameExtractor


def test_sampling_modes():
    """Test nativeFPS, fixedFPS, and uniform sampling modes."""
    extractor_fixed = FrameExtractor(sampling_mode="fixedFPS", target_fps=25.0, max_sequence_length=50)
    res_fixed = extractor_fixed.extract_frames_from_video("dummy_video.mp4")
    assert res_fixed["frames"].shape == (50, 224, 224, 3)
    assert res_fixed["attention_mask"].shape == (50,)
    assert len(res_fixed["timestamps_sec"]) <= 50

    extractor_uniform = FrameExtractor(sampling_mode="uniform", num_uniform_frames=32, max_sequence_length=50)
    res_uniform = extractor_uniform.extract_frames_from_video("dummy_video.mp4")
    assert res_uniform["valid_length"] == 32
    assert res_uniform["attention_mask"][:32].all()
    assert not res_uniform["attention_mask"][32:].any()


def test_spatial_transforms_and_crop():
    """Test center crop and signer-centric crop spatial transformations."""
    extractor_center = FrameExtractor(target_size=(112, 112), crop_mode="center")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cropped = extractor_center.apply_spatial_transforms(dummy_frame)
    assert cropped.shape == (112, 112, 3)

    extractor_signer = FrameExtractor(target_size=(224, 224), crop_mode="signer_centric")
    bbox = (200, 100, 600, 500)
    cropped_signer = extractor_signer.apply_spatial_transforms(dummy_frame, bbox_signer=bbox)
    assert cropped_signer.shape == (224, 224, 3)


def test_padding_and_attention_mask():
    """Test sequence padding to max_sequence_length and boolean attention mask."""
    extractor = FrameExtractor(max_sequence_length=100, target_size=(224, 224))
    dummy_frames = [np.ones((224, 224, 3), dtype=np.uint8) for _ in range(15)]

    padded, mask = extractor.apply_padding_and_mask(dummy_frames)
    assert padded.shape == (100, 224, 224, 3)
    assert mask.shape == (100,)
    assert mask[:15].all()
    assert not mask[15:].any()


def test_timestamp_preservation():
    """Test timestamp in seconds preservation for extracted frames."""
    extractor = FrameExtractor(sampling_mode="fixedFPS", target_fps=10.0, max_sequence_length=10)
    res = extractor.extract_frames_from_video("dummy_video.mp4")

    timestamps = res["timestamps_sec"]
    assert len(timestamps) > 0
    assert timestamps[0] == 0.0
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]


def test_throughput_benchmarking():
    """Test preprocessing throughput benchmarking."""
    extractor = FrameExtractor(max_sequence_length=32)
    bench_results = extractor.benchmark_throughput(num_samples=10)

    assert bench_results["num_samples"] == 10
    assert bench_results["samples_per_second"] > 0.0
    assert bench_results["frames_per_second"] > 0.0
