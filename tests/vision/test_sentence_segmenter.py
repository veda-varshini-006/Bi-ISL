"""Tests for SignActivityDetector continuous sign segmentation."""

import pytest
import numpy as np
from src.vision.sentence_segmenter import SignActivityDetector


def test_sign_activity_detector_static_input():
    detector = SignActivityDetector(
        motion_threshold=0.015,
        rest_duration_frames=5,
        min_sentence_frames=10
    )

    static_frame = np.ones((75, 3), dtype=np.float32) * 0.5
    for _ in range(15):
        is_signing, completed_seq = detector.process_frame(static_frame)
        assert not is_signing
        assert completed_seq is None


def test_sign_activity_detector_dynamic_signing_and_pause():
    detector = SignActivityDetector(
        motion_threshold=0.01,
        rest_duration_frames=3,
        min_sentence_frames=5
    )

    # Establish previous frame
    detector.process_frame(np.zeros((75, 3), dtype=np.float32))

    # Simulate dynamic movement for 8 frames
    for i in range(8):
        dynamic_frame = np.random.randn(75, 3).astype(np.float32) * 2.0
        is_signing, completed_seq = detector.process_frame(dynamic_frame)
        assert is_signing

    # Simulate rest pause for 4 frames
    static_frame = np.zeros((75, 3), dtype=np.float32)
    completed = None
    for _ in range(4):
        is_signing, seq = detector.process_frame(static_frame)
        if seq is not None:
            completed = seq

    assert completed is not None
    assert len(completed) >= 5


def test_sign_activity_detector_reset():
    detector = SignActivityDetector()
    detector.process_frame(np.zeros((75, 3), dtype=np.float32))
    detector.process_frame(np.ones((75, 3), dtype=np.float32) * 2.0)
    assert detector.current_buffer_length > 0

    detector.reset()
    assert detector.current_buffer_length == 0
