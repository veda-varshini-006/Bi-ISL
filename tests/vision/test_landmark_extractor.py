"""Unit tests for Bi-ISL Multimodal MediaPipe Landmark Extractor and Normalizer."""

import pytest
import numpy as np

from src.vision.landmark_extractor import LandmarkExtractor, MultimodalPoseSample
from src.vision.normalizer import LandmarkNormalizer


def test_multimodal_landmark_extraction_schema():
    """Test multimodal landmark extraction schema and fields."""
    extractor = LandmarkExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample = extractor.extract_landmarks(dummy_frame, frame_index=5, timestamp_sec=0.167)
    assert sample.frame_index == 5
    assert sample.timestamp_sec == 0.167
    assert len(sample.pose_landmarks) == 33
    assert len(sample.left_hand_landmarks) == 21
    assert len(sample.right_hand_landmarks) == 21
    assert len(sample.face_landmarks) == 468
    assert "jawOpen" in sample.face_blendshapes


def test_explicit_missing_modality_masks():
    """Test explicit missing modality masks and non-interpolation flag."""
    sample = MultimodalPoseSample(
        frame_index=0,
        timestamp_sec=0.0,
        mask_pose=True,
        mask_left_hand=False,  # Left hand occluded / missing
        mask_right_hand=True,
        mask_face=True,
        is_interpolated=False
    )

    assert sample.mask_left_hand is False
    assert sample.is_interpolated is False


def test_body_reference_normalization():
    """Test landmark normalization relative to shoulder midpoint origin and scale."""
    normalizer = LandmarkNormalizer()

    pose_kps = np.zeros((33, 4), dtype=np.float32)
    # Left shoulder [0.4, 0.4, 0.0], Right shoulder [0.6, 0.4, 0.0]
    # Midpoint = [0.5, 0.4, 0.0], Shoulder dist = 0.2
    pose_kps[11] = [0.4, 0.4, 0.0, 1.0]
    pose_kps[12] = [0.6, 0.4, 0.0, 1.0]

    sample = MultimodalPoseSample(
        frame_index=0,
        timestamp_sec=0.0,
        mask_pose=True,
        pose_landmarks=pose_kps.tolist()
    )

    norm_sample = normalizer.normalize_sample(sample)
    norm_pose = np.array(norm_sample.pose_landmarks)

    # Normalized left shoulder: (0.4 - 0.5) / 0.2 = -0.5
    assert pytest.approx(norm_pose[11, 0], abs=1e-3) == -0.5
    # Normalized right shoulder: (0.6 - 0.5) / 0.2 = +0.5
    assert pytest.approx(norm_pose[12, 0], abs=1e-3) == 0.5
    # Y midpoint: (0.4 - 0.4) / 0.2 = 0.0
    assert pytest.approx(norm_pose[11, 1], abs=1e-3) == 0.0


def test_debug_overlay_rendering():
    """Test visual debug overlay rendering."""
    extractor = LandmarkExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample = extractor.extract_landmarks(dummy_frame, frame_index=1, timestamp_sec=0.033)
    overlay = extractor.render_debug_overlay(dummy_frame, sample)

    assert overlay.shape == dummy_frame.shape
