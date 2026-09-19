"""Unit and integration tests for Facial / Non-Manual Marker (NMM) Controller (Prompt 76).

Tests:
1. NonManualTagSpec AHDR envelope weight calculations
2. Eyebrow state transitions (FURROWED, RAISED, NEUTRAL)
3. 32 FACS blendshape scheduling
4. Head pose channels (nod pitch, shake yaw, tilt roll)
5. Body lean channels (forward lean, backward lean)
6. Keyframe timing synchronization
7. NMMDebugOverlay HUD formatting and active tag tracking
8. NMMBlendshapeEngine computation
9. RetargetingMotionMapper NMM tag integration
"""

import math
import pytest

from src.avatar.animation_clip_library import AnimationClipLibrary, AnimationClipKeyframe
from src.avatar.nmm_controller import NMMController, NonManualTagSpec, NMMFrameState
from src.avatar.nmm_debug_overlay import NMMDebugOverlay
from src.avatar.blendshape_engine import NMMBlendshapeEngine
from src.avatar.motion_mapper import RetargetingMotionMapper


def test_tag_spec_ahdr_envelope():
    spec = NonManualTagSpec(
        tag="eyebrows_furrowed",
        start_ms=100.0,
        end_ms=500.0,
        intensity=1.0,
        attack_ms=100.0,
        decay_ms=100.0
    )

    assert spec.get_envelope_weight(50.0) == 0.0
    assert spec.get_envelope_weight(550.0) == 0.0
    assert abs(spec.get_envelope_weight(100.0) - 0.0) < 1e-4
    assert spec.get_envelope_weight(300.0) == 1.0  # Peak hold phase
    assert 0.0 < spec.get_envelope_weight(150.0) < 1.0  # Attack phase
    assert 0.0 < spec.get_envelope_weight(450.0) < 1.0  # Decay phase


def test_nmm_controller_eyebrow_states():
    controller = NMMController()

    furrowed_spec = NonManualTagSpec(tag="eyebrows_furrowed", start_ms=0.0, end_ms=1000.0)
    state_furrowed = controller.evaluate_nmm_state([furrowed_spec], 500.0)
    assert state_furrowed.eyebrow_state == "FURROWED"
    assert state_furrowed.blendshapes["browDownLeft"] > 0.0
    assert state_furrowed.blendshapes["browDownRight"] > 0.0

    raised_spec = NonManualTagSpec(tag="eyebrows_raised", start_ms=0.0, end_ms=1000.0)
    state_raised = controller.evaluate_nmm_state([raised_spec], 500.0)
    assert state_raised.eyebrow_state == "RAISED"
    assert state_raised.blendshapes["browOuterUpLeft"] > 0.0
    assert state_raised.blendshapes["browOuterUpRight"] > 0.0

    state_neutral = controller.evaluate_nmm_state([], 500.0)
    assert state_neutral.eyebrow_state == "NEUTRAL"


def test_nmm_controller_head_pose_and_body_lean():
    controller = NMMController()

    specs = [
        NonManualTagSpec(tag="head_nod_slight", start_ms=0.0, end_ms=1000.0),
        NonManualTagSpec(tag="head_shake_negation", start_ms=0.0, end_ms=1000.0),
        NonManualTagSpec(tag="body_lean_forward", start_ms=0.0, end_ms=1000.0)
    ]

    state = controller.evaluate_nmm_state(specs, 500.0)

    assert "head_nod_slight" in state.active_tags
    assert "head_shake_negation" in state.active_tags
    assert "body_lean_forward" in state.active_tags

    # Body pitch forward lean should be positive (> 0 rad)
    assert state.body_lean_euler[0] > 0.0


def test_keyframe_timing_synchronization():
    controller = NMMController()

    kf1 = AnimationClipKeyframe(timestamp_ms=0.0, joint_rotations={"Head": [0.0, 0.0, 0.0, 1.0]})
    kf2 = AnimationClipKeyframe(timestamp_ms=500.0, joint_rotations={"Head": [0.0, 0.0, 0.0, 1.0]})

    specs = [NonManualTagSpec(tag="eyebrows_furrowed", start_ms=200.0, end_ms=800.0)]

    synced_kfs = controller.synchronize_nmm_to_keyframes([kf1, kf2], specs)

    assert len(synced_kfs) == 2
    assert synced_kfs[0].blendshape_weights.get("browDownLeft", 0.0) == 0.0
    assert synced_kfs[1].blendshape_weights.get("browDownLeft", 0.0) > 0.0


def test_nmm_debug_overlay_formatting():
    state = NMMFrameState(
        timestamp_ms=250.0,
        active_tags=["eyebrows_furrowed", "body_lean_forward"],
        eyebrow_state="FURROWED",
        blendshapes={"browDownLeft": 0.85, "browDownRight": 0.85},
        head_pose_euler=[0.1, 0.0, 0.0],
        body_lean_euler=[0.26, 0.0, 0.0]
    )

    hud_text = NMMDebugOverlay.format_hud_text(state, frame_index=15)
    assert "ISL AVATAR NMM DEBUG OVERLAY" in hud_text
    assert "eyebrows_furrowed" in hud_text
    assert "FURROWED" in hud_text
    assert "Timestamp   : 250.0 ms" in hud_text

    frame_data = NMMDebugOverlay.generate_overlay_frame_data(state, frame_index=15)
    assert frame_data["frame_index"] == 15
    assert frame_data["eyebrow_state"] == "FURROWED"
    assert frame_data["active_tag_count"] == 2


def test_nmm_blendshape_engine():
    engine = NMMBlendshapeEngine()
    tags = [
        {"tag": "eyebrows_furrowed", "start_ms": 0.0, "end_ms": 1000.0},
        {"tag": "squint_intense", "start_ms": 0.0, "end_ms": 1000.0}
    ]

    blendshapes = engine.compute_facial_blendshapes(tags)
    assert isinstance(blendshapes, dict)
    assert len(blendshapes) == 32
    assert blendshapes["browDownLeft"] > 0.0
    assert blendshapes["eyeSquintLeft"] > 0.0


def test_motion_mapper_nmm_integration():
    mapper = RetargetingMotionMapper()

    isl_ir = {
        "gloss": "DOCTOR",
        "non_manual_markers": [
            {"tag": "eyebrows_furrowed", "start_ms": 0.0, "end_ms": 1000.0},
            {"tag": "body_lean_forward", "start_ms": 0.0, "end_ms": 1000.0}
        ]
    }

    keyframes = mapper.map_to_motion_keyframes(isl_ir)
    assert len(keyframes) > 0
    first_kf = keyframes[0]
    assert "browDownLeft" in first_kf["blendshape_weights"]
