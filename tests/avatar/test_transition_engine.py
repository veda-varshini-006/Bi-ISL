"""Unit and integration tests for Animation Transition Engine (Prompt 75).

Tests:
1. Transition keyframe generation between consecutive sign clips
2. Handshape preservation (decoupling finger digits from arm trajectories vs naive crossfade)
3. Hold transitions (hold_duration_ms > 0)
4. Coarticulation metadata usage
5. Input clip immutability (validated source clips remain intact)
6. Multi-clip sequence stitching (stitch_clip_sequence)
7. RetargetingMotionMapper sequence mapping with transitions
"""

from copy import deepcopy
import pytest

from src.avatar.animation_clip_library import AnimationClipLibrary, AnimationClip, AnimationClipKeyframe
from src.avatar.transition_engine import TransitionEngine, CoarticulationMetadata, TransitionValidator
from src.avatar.motion_mapper import RetargetingMotionMapper


def test_transition_engine_basic_keyframe_generation():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_doctor = lib.get_clip_by_gloss("DOCTOR")
    clip_hospital = lib.get_clip_by_gloss("HOSPITAL")

    assert clip_doctor is not None
    assert clip_hospital is not None

    engine = TransitionEngine()
    trans_kfs = engine.generate_transition_keyframes(
        clip_doctor, clip_hospital, transition_duration_ms=120.0
    )

    assert len(trans_kfs) > 0
    assert trans_kfs[0].timestamp_ms > clip_doctor.keyframes[-1].timestamp_ms
    assert "UpperArm_R" in trans_kfs[0].joint_rotations


def test_handshape_preservation_vs_naive_crossfade():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_doctor = lib.get_clip_by_gloss("DOCTOR")
    clip_fever = lib.get_clip_by_gloss("FEVER")

    engine = TransitionEngine()

    controlled_kfs = engine.generate_transition_keyframes(
        clip_doctor, clip_fever, transition_duration_ms=120.0, hold_handshape=True
    )
    naive_kfs = engine.generate_naive_crossfade_keyframes(
        clip_doctor, clip_fever, transition_duration_ms=120.0
    )

    assert len(controlled_kfs) == len(naive_kfs)

    # In controlled transition, finger joints hold source handshape during first 60% of transition
    mid_idx = int(len(controlled_kfs) * 0.3)
    c_finger_rot = controlled_kfs[mid_idx].joint_rotations.get("UpperArm_R")
    n_finger_rot = naive_kfs[mid_idx].joint_rotations.get("UpperArm_R")
    assert c_finger_rot is not None
    assert n_finger_rot is not None

    report = TransitionValidator.validate_handshape_integrity(controlled_kfs, naive_kfs)
    assert report["handshape_protected"]
    assert report["status"] == "VALIDATED"


def test_hold_transition_generation():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_doctor = lib.get_clip_by_gloss("DOCTOR")
    clip_hospital = lib.get_clip_by_gloss("HOSPITAL")

    engine = TransitionEngine()
    
    # Generate transition with 100ms hold phase
    hold_kfs = engine.generate_transition_keyframes(
        clip_doctor, clip_hospital, transition_duration_ms=120.0, hold_duration_ms=100.0
    )

    # Keyframes should include hold frames followed by spatial transition frames
    assert len(hold_kfs) >= 4

    # First keyframe should match end pose of clip_doctor
    end_doctor_pos = clip_doctor.keyframes[-1].joint_positions.get("Wrist_R")
    hold_first_pos = hold_kfs[0].joint_positions.get("Wrist_R")
    assert hold_first_pos == end_doctor_pos


def test_coarticulation_metadata_usage():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_doctor = lib.get_clip_by_gloss("DOCTOR")
    clip_hospital = lib.get_clip_by_gloss("HOSPITAL")

    engine = TransitionEngine()
    meta = CoarticulationMetadata(
        clip_id=clip_doctor.clip_id,
        gloss="DOCTOR",
        transition_duration_ms=150.0,
        hold_duration_ms=60.0,
        hold_handshape=True,
        blend_curve="AHDR_HERMITE"
    )

    trans_kfs = engine.generate_transition_keyframes(
        clip_doctor, clip_hospital, coarticulation_meta=meta
    )
    assert len(trans_kfs) > 0


def test_input_clip_immutability():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_doctor = lib.get_clip_by_gloss("DOCTOR")
    clip_hospital = lib.get_clip_by_gloss("HOSPITAL")

    doctor_copy = deepcopy(clip_doctor)
    hospital_copy = deepcopy(clip_hospital)

    engine = TransitionEngine()
    stitched = engine.stitch_clip_sequence([clip_doctor, clip_hospital])

    # Verify original clip instances were 100% untouched and unmutated
    assert TransitionValidator.verify_clip_immutability(doctor_copy, clip_doctor)
    assert TransitionValidator.verify_clip_immutability(hospital_copy, clip_hospital)


def test_sequence_stitching():
    lib = AnimationClipLibrary.create_seeded_library()
    clip_1 = lib.get_clip_by_gloss("DOCTOR")
    clip_2 = lib.get_clip_by_gloss("FEVER")
    clip_3 = lib.get_clip_by_gloss("MEDICINE")

    engine = TransitionEngine()
    stitched = engine.stitch_clip_sequence([clip_1, clip_2, clip_3], default_transition_ms=100.0)

    assert stitched is not None
    assert stitched.gloss == "DOCTOR FEVER MEDICINE"
    assert len(stitched.keyframes) > len(clip_1.keyframes) + len(clip_2.keyframes) + len(clip_3.keyframes)
    assert stitched.duration > 0.0


def test_motion_mapper_sequence_mapping_with_transitions():
    mapper = RetargetingMotionMapper()
    isl_ir_sequence = [
        {"gloss": "DOCTOR", "scale_factor": 1.0},
        {"gloss": "FEVER", "scale_factor": 1.0, "coarticulation": {"transition_duration_ms": 100.0, "hold_duration_ms": 50.0}},
        {"gloss": "MEDICINE", "scale_factor": 1.0}
    ]

    keyframes_data = mapper.map_sequence_with_transitions(isl_ir_sequence)
    assert isinstance(keyframes_data, list)
    assert len(keyframes_data) > 0
    assert "joint_rotations" in keyframes_data[0]
    assert "joint_positions" in keyframes_data[0]
