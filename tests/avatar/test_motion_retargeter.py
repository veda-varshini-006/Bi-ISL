"""Unit and integration tests for Motion Retargeting Subsystem (Prompt 74).

Tests:
1. HumanoidRigConfig joint limit specification
2. MotionRetargeter keyframe and clip retargeting
3. Finger alignment verification
4. Wrist orientation verification
5. Joint constraints verification
6. Body scale verification
7. Handedness mirroring verification
8. RetargetingMotionMapper IR token mapping
9. RetargetingVerifier 5-point verification suite
10. VisualRegressionEngine benchmark scenes and suite execution
"""

import math
import os
import tempfile
import pytest

from src.avatar.animation_clip_library import AnimationClipLibrary, AnimationClip, AnimationClipKeyframe
from src.avatar.motion_retargeter import HumanoidRigConfig, MotionRetargeter, VectorMath, JointLimit
from src.avatar.motion_mapper import RetargetingMotionMapper
from src.avatar.retargeting_verifier import RetargetingVerifier, RetargetingVerificationReport
from src.avatar.visual_regression_scenes import VisualRegressionEngine, VisualRegressionSceneConfig


def test_humanoid_rig_config_defaults():
    config = HumanoidRigConfig()
    assert config.rig_name == "UnityHumanoidRig_Standard"
    assert config.height_meters == 1.75
    assert len(config.bones) == 67
    assert "Hips" in config.bones
    assert "Wrist_R" in config.bones
    assert "Index_01_L" in config.bones
    assert "Wrist_R" in config.joint_limits


def test_vector_math_operations():
    # Test quaternion normalization
    q = [0.1, 0.2, 0.3, 0.4]
    norm_q = VectorMath.normalize_quat(q)
    assert abs(VectorMath.norm(norm_q) - 1.0) < 1e-6

    # Test Euler <-> Quaternion roundtrip
    pitch, yaw, roll = 0.1, 0.2, -0.15
    q_conv = VectorMath.euler_to_quat(pitch, yaw, roll)
    p_out, y_out, r_out = VectorMath.quat_to_euler(q_conv)
    assert abs(pitch - p_out) < 1e-4
    assert abs(yaw - y_out) < 1e-4
    assert abs(roll - r_out) < 1e-4

    # Test matrix orthonormalization
    p = [0.0, 1.0, 0.0]
    d = [0.0, 0.0, 1.0]
    R = VectorMath.matrix_from_vectors(p, d)
    q_mat = VectorMath.matrix_to_quat(R)
    assert len(q_mat) == 4
    assert abs(VectorMath.norm(q_mat) - 1.0) < 1e-6


def test_motion_retargeter_single_keyframe():
    retargeter = MotionRetargeter()
    
    kf = AnimationClipKeyframe(
        timestamp_ms=100.0,
        joint_rotations={
            "UpperArm_R": [0.1, 0.2, 0.0, 1.0],
            "Wrist_R": [0.05, 0.0, 0.0, 1.0],
            "Index_01_R": [0.5, 0.1, 0.0, 1.0]  # Valid finger angle
        },
        joint_positions={
            "Wrist_R": [0.2, 1.2, 0.4]
        },
        blendshape_weights={"browDownLeft": 0.5}
    )

    retargeted_kf = retargeter.retarget_keyframe(kf, scale_factor=1.2)
    
    # Position should be scaled by 1.2
    assert retargeted_kf.joint_positions["Wrist_R"] == [0.2 * 1.2, 1.2 * 1.2, 0.4 * 1.2]
    assert "UpperArm_R" in retargeted_kf.joint_rotations
    assert len(retargeted_kf.joint_rotations["UpperArm_R"]) == 4


def test_motion_retargeter_joint_clamping():
    retargeter = MotionRetargeter()
    
    # Create keyframe with out-of-bounds joint angle (e.g. elbow flex > 145 deg -> 180 deg)
    out_of_bounds_quat = VectorMath.euler_to_quat(math.radians(180), 0.0, 0.0)
    kf = AnimationClipKeyframe(
        timestamp_ms=0.0,
        joint_rotations={"LowerArm_R": out_of_bounds_quat},
        joint_positions={"Wrist_R": [0.15, 1.0, 0.3]}
    )

    retargeted_kf = retargeter.retarget_keyframe(kf)
    clamped_q = retargeted_kf.joint_rotations["LowerArm_R"]
    pitch, _, _ = VectorMath.quat_to_euler(clamped_q)

    # Pitch should be clamped to max_pitch (145 degrees = ~2.53 rad)
    assert pitch <= math.radians(145) + 1e-3


def test_motion_retargeter_handedness_mirroring():
    retargeter = MotionRetargeter()
    lib = AnimationClipLibrary.create_seeded_library()
    right_clip = lib.get_clip_by_gloss("DOCTOR")
    assert right_clip is not None

    # Retarget right-handed clip to left-handed
    left_retargeted = retargeter.retarget_clip(right_clip, target_handedness="LEFT")
    assert left_retargeted.dominant_hand == "LEFT"

    # Verify positions X coordinate reflected (x -> -x)
    first_kf = left_retargeted.keyframes[0]
    assert "Wrist_L" in first_kf.joint_positions
    assert first_kf.joint_positions["Wrist_L"][0] < 0  # Was positive for Wrist_R


def test_retargeting_motion_mapper():
    mapper = RetargetingMotionMapper()
    isl_ir = {
        "gloss": "DOCTOR",
        "scale_factor": 1.1,
        "handedness": "RIGHT"
    }
    keyframes_data = mapper.map_to_motion_keyframes(isl_ir)
    assert isinstance(keyframes_data, list)
    assert len(keyframes_data) > 0
    assert "joint_rotations" in keyframes_data[0]
    assert "joint_positions" in keyframes_data[0]


def test_retargeting_verifier_5_point_checks():
    lib = AnimationClipLibrary.create_seeded_library()
    src_clip = lib.get_clip_by_gloss("HELP")
    assert src_clip is not None

    retargeter = MotionRetargeter()
    retargeted_clip = retargeter.retarget_clip(src_clip, scale_factor=1.0)

    verifier = RetargetingVerifier()
    report = verifier.verify_all(src_clip, retargeted_clip, scale_factor=1.0)

    assert isinstance(report, RetargetingVerificationReport)
    assert report.is_valid
    assert report.finger_alignment_valid
    assert report.wrist_orientation_valid
    assert report.joint_constraints_valid
    assert report.body_scale_valid
    assert report.handedness_valid
    assert len(report.errors) == 0


def test_visual_regression_engine():
    lib = AnimationClipLibrary.create_seeded_library()
    engine = VisualRegressionEngine()

    benchmark_scenes = engine.create_benchmark_scenes()
    assert len(benchmark_scenes) == 5

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifests = engine.generate_scene_manifests(tmp_dir)
        assert len(manifests) == 5
        for m in manifests:
            assert m.exists()

        summary = engine.run_visual_regression_suite(lib, output_dir=tmp_dir)
        assert summary["overall_passed"]
        assert summary["total_scenes_evaluated"] == 5
        assert len(summary["scenes"]) == 5

        report_file = os.path.join(tmp_dir, "visual_regression_report.json")
        assert os.path.exists(report_file)
