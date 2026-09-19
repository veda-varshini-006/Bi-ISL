"""Unit and integration tests for Unity Controller & API Subsystem (Prompt 77).

Tests:
1. UnityISLAnimationPayload JSON serialization and deserialization
2. UnityPayloadExporter payload generation from ISL IR
3. DeterministicAvatarRenderer playback controls (play, pause, repeat, speed, step-through, seek, debug_mode)
4. Verification that Unity payload schema contains zero translation logic
"""

import json
import os
import tempfile
import pytest

from src.avatar.unity_payload_exporter import UnityISLAnimationPayload, UnityPayloadExporter
from src.avatar.avatar_renderer import DeterministicAvatarRenderer


def test_unity_payload_serialization():
    payload = UnityISLAnimationPayload(
        sequence_id="SEQ_TEST_01",
        gloss_sequence=["DOCTOR", "FEVER"],
        total_duration_sec=2.5,
        fps=60.0,
        total_frames=150,
        keyframes=[{"timestamp_ms": 0.0, "joint_rotations": {}, "joint_positions": {}}],
        metadata={"zero_unity_translation": True}
    )

    json_str = payload.to_json()
    assert "SEQ_TEST_01" in json_str
    assert "DOCTOR" in json_str

    deserialized = UnityISLAnimationPayload.from_dict(json.loads(json_str))
    assert deserialized.sequence_id == "SEQ_TEST_01"
    assert deserialized.total_frames == 150
    assert deserialized.metadata["zero_unity_translation"] is True


def test_unity_payload_exporter_from_ir_sequence():
    exporter = UnityPayloadExporter()
    isl_ir_sequence = [
        {"gloss": "DOCTOR", "scale_factor": 1.0},
        {"gloss": "FEVER", "scale_factor": 1.0, "non_manual_markers": [{"tag": "eyebrows_furrowed"}]}
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = os.path.join(tmp_dir, "test_payload.json")
        payload = exporter.export_payload_from_ir_sequence(isl_ir_sequence, output_json_path=json_file)

        assert os.path.exists(json_file)
        assert payload.sequence_id is not None
        assert len(payload.keyframes) > 0
        assert payload.metadata["zero_unity_translation"] is True

        # First keyframe should contain pre-computed nmm_state and debug_overlay
        first_kf = payload.keyframes[0]
        assert "nmm_state" in first_kf
        assert "debug_overlay" in first_kf


def test_deterministic_avatar_renderer_playback_api():
    exporter = UnityPayloadExporter()
    isl_ir_sequence = [{"gloss": "HELP"}]
    payload = exporter.export_payload_from_ir_sequence(isl_ir_sequence)

    renderer = DeterministicAvatarRenderer()
    renderer.load_payload(payload)

    assert renderer.current_frame_index == 0
    assert not renderer.is_playing

    # Play & Pause
    renderer.play()
    assert renderer.is_playing
    renderer.pause()
    assert not renderer.is_playing

    # Repeat & Speed
    renderer.set_repeat(True)
    assert renderer.is_repeat_enabled
    renderer.set_speed(1.5)
    assert renderer.playback_speed == 1.5

    # Step Forward
    kf1 = renderer.step_forward()
    assert renderer.current_frame_index == 1
    assert "timestamp_ms" in kf1

    # Seek To Frame
    kf_seek = renderer.seek_to_frame(5)
    assert renderer.current_frame_index == 5

    # Step Backward
    kf_back = renderer.step_backward()
    assert renderer.current_frame_index == 4

    # Debug Mode
    renderer.set_debug_mode(True)
    assert renderer.is_debug_mode_enabled


def test_zero_unity_translation_logic_schema_isolation():
    exporter = UnityPayloadExporter()
    isl_ir_sequence = [{"gloss": "DOCTOR"}, {"gloss": "HELP"}]
    payload = exporter.export_payload_from_ir_sequence(isl_ir_sequence)

    payload_dict = payload.to_dict()
    
    # Assert payload contains pre-evaluated keyframe matrices only, zero grammar/translation modules
    assert "grammar" not in payload_dict
    assert "translation_rules" not in payload_dict
    assert payload_dict["metadata"]["zero_unity_translation"] is True
