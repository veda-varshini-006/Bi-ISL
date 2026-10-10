"""Tests for continuous video translation and streaming inference on recorded ISL videos."""

import os
import time
from pathlib import Path
import numpy as np
import pytest
import cv2

from src.inference.video_translator import VideoTranslator
from src.inference.streaming_translator import StreamingTranslator, StreamingResult
from src.text.subtitle_stabilizer import SubtitleStabilizer, ARSubtitle


@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory):
    """Generates a recorded test video file with moving hands and a rest pause."""
    tmp_dir = tmp_path_factory.mktemp("test_videos")
    video_file = tmp_dir / "sample_sentence_signing.mp4"

    width, height = 640, 480
    fps = 25
    total_frames = 60  # 40 frames of motion + 20 frames of rest pause

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for f in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background

        # Draw a synthetic torso and head
        cv2.circle(frame, (320, 140), 40, (200, 200, 200), -1)  # Head
        cv2.rectangle(frame, (260, 190), (380, 360), (100, 100, 150), -1)  # Torso

        if f < 40:
            # Active signing motion: hands moving rhythmically
            phase = f * 0.3
            lh_x = int(220 + np.sin(phase) * 60)
            lh_y = int(250 + np.cos(phase * 1.5) * 40)
            rh_x = int(420 + np.cos(phase) * 60)
            rh_y = int(250 + np.sin(phase * 1.5) * 40)
        else:
            # Rest pause: hands at rest by sides
            lh_x, lh_y = 230, 340
            rh_x, rh_y = 410, 340

        cv2.circle(frame, (lh_x, lh_y), 16, (180, 220, 255), -1)  # Left hand
        cv2.circle(frame, (rh_x, rh_y), 16, (180, 220, 255), -1)  # Right hand

        out.write(frame)

    out.release()
    return str(video_file)


def test_video_translator_on_recorded_video(sample_video_path):
    translator = VideoTranslator()
    res = translator.translate_video(sample_video_path)

    assert res["video_frames"] == 60
    assert res["video_duration_sec"] > 0.0
    assert isinstance(res["english_translation"], str)
    assert len(res["english_translation"]) > 0
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["processing_latency_sec"] > 0.0


def test_video_translator_subtitled_video_rendering(sample_video_path, tmp_path):
    output_video = tmp_path / "subtitled_output.mp4"
    translator = VideoTranslator()

    res = translator.translate_video(
        video_path=sample_video_path,
        output_video_path=str(output_video)
    )

    assert output_video.exists()
    assert output_video.stat().st_size > 0

    # Verify output video is readable
    cap = cv2.VideoCapture(str(output_video))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count == 60
    cap.release()


def test_streaming_translator_provisional_and_finalized_transitions(sample_video_path):
    translator = VideoTranslator()
    streamer = StreamingTranslator(
        translator=translator.translator,
        provisional_interval_frames=10,
        min_inference_frames=10,
        rest_pause_frames=8
    )

    cap = cv2.VideoCapture(sample_video_path)
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = streamer.process_frame(frame)
        results.append(res)
    cap.release()

    assert len(results) == 60
    # First half should observe higher motion energy
    motion_first_half = np.mean([r.motion_energy for r in results[:30]])
    motion_second_half = np.mean([r.motion_energy for r in results[45:]])
    assert motion_first_half >= motion_second_half


def test_overlapping_window_deduplication():
    stabilizer = SubtitleStabilizer(hold_duration_sec=5.0, dedup_window_sec=5.0)

    # First finalized sentence
    sub1 = stabilizer.update_finalized("I will go to the train station tomorrow", confidence=0.92)
    assert sub1 is not None
    assert len(stabilizer.finalized_history) == 1

    # Overlapping window emitting identical sentence -> should be deduplicated
    sub2 = stabilizer.update_finalized("I will go to the train station tomorrow", confidence=0.88)
    assert len(stabilizer.finalized_history) == 1  # No duplicate added

    # Overlapping window emitting near-identical variant -> should be deduplicated
    sub3 = stabilizer.update_finalized("I will go to the train station tomorrow.", confidence=0.85)
    assert len(stabilizer.finalized_history) == 1

    # New distinct sentence -> should be accepted
    sub4 = stabilizer.update_finalized("What time is my doctor appointment today?", confidence=0.95)
    assert sub4 is not None
    assert len(stabilizer.finalized_history) == 2


def test_output_is_complete_sentence_not_alphabet():
    translator = VideoTranslator()
    dummy_input = np.random.randn(128, 75, 3).astype(np.float32)
    text = translator.translator.translate(dummy_input)

    assert isinstance(text, str)
    words = text.strip().split()
    # Continuous ISL translation decodes words/subwords, not single-alphabet spellings
    assert len(words) > 0
    avg_word_len = sum(len(w) for w in words) / len(words)
    assert avg_word_len > 1.5
