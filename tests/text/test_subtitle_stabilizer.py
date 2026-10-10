"""Tests for SubtitleStabilizer and ARSubtitle smoothing."""

import time
import pytest
from src.text.subtitle_stabilizer import SubtitleStabilizer, ARSubtitle


def test_subtitle_stabilizer_initial_update():
    stabilizer = SubtitleStabilizer(hold_duration_sec=2.0, min_confidence=0.5)
    res = stabilizer.update("Hello how are you", confidence=0.85)

    assert res is not None
    assert res.text == "Hello how are you"
    assert res.confidence == 0.85
    assert not res.is_expired


def test_subtitle_stabilizer_low_confidence_filtering():
    stabilizer = SubtitleStabilizer(min_confidence=0.7)
    res = stabilizer.update("Low confidence prediction", confidence=0.4)
    assert res is None


def test_subtitle_stabilizer_smoothing():
    stabilizer = SubtitleStabilizer(similarity_threshold=0.8, hold_duration_sec=3.0)
    res1 = stabilizer.update("Where is the hospital", confidence=0.9)
    assert res1.text == "Where is the hospital"

    # High similarity prediction refreshes active subtitle
    res2 = stabilizer.update("Where is the hospital", confidence=0.85)
    assert res2 is not None
    assert res2.text == "Where is the hospital"
    assert not res2.is_expired


def test_subtitle_stabilizer_clear():
    stabilizer = SubtitleStabilizer()
    stabilizer.update("Testing clear", confidence=0.9)
    assert stabilizer.get_current_subtitle() is not None

    stabilizer.clear()
    assert stabilizer.get_current_subtitle() is None
