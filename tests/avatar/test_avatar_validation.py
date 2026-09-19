"""Tests for Avatar Validation Suite (Prompt 78)."""

import os
import tempfile
import pytest

from src.avatar.avatar_validation_suite import AvatarValidationSuite
from src.avatar.motion_mapper import RetargetingMotionMapper
from src.avatar.animation_clip_library import AnimationClip, AnimationClipKeyframe

def test_avatar_validation_suite_golden_path():
    suite = AvatarValidationSuite()
    mapper = RetargetingMotionMapper()
    
    ir_seq = [
        {"gloss": "DOCTOR"},
        {"gloss": "HELP", "non_manual_markers": [{"tag": "furrowed"}]}
    ]
    
    # We construct a dummy clip manually with the expected NMM and finger joints
    # because generating the actual stitched one might be complex here.
    lib = suite.clip_library
    c1 = lib.get_clip_by_gloss("DOCTOR")
    c2 = lib.get_clip_by_gloss("HELP")
    assert c1 is not None and c2 is not None
    
    # Instead of full mapping, let's just make sure all bones are present in a keyframe
    prefixes = ["Thumb_", "Index_", "Middle_", "Ring_", "Pinky_"]
    suffixes = ["01", "02", "03"]
    sides = ["L", "R"]
    required_fingers = [f"{p}{s}_{side}" for p in prefixes for s in suffixes for side in sides]
    
    base_kf = c1.keyframes[0]
    new_rotations = dict(base_kf.joint_rotations)
    for bf in required_fingers:
        new_rotations[bf] = [0.0, 0.0, 0.0, 1.0]
        
    dummy_kf = AnimationClipKeyframe(
        timestamp_ms=0.0,
        joint_rotations=new_rotations,
        joint_positions=dict(base_kf.joint_positions),
        blendshape_weights={"browDownLeft": 0.5}
    )
    
    dummy_kf2 = AnimationClipKeyframe(
        timestamp_ms=16.0,
        joint_rotations=new_rotations,
        joint_positions=dict(base_kf.joint_positions),
        blendshape_weights={"browDownLeft": 0.5}
    )
    
    stitched = AnimationClip(
        clip_id="test_stitch", sign_id="test", gloss="test", duration=1.0, 
        dominant_hand="RIGHT", keyframes=[dummy_kf, dummy_kf2]
    )
    
    report = suite.validate_sequence(ir_seq, stitched)
    
    assert report.is_valid
    for e in report.entries:
        assert e.passed, f"Check {e.check_name} failed unexpectedly: {e.message}"
        
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "report.json")
        suite.export_validation_report(report, report_path)
        assert os.path.exists(report_path)

def test_avatar_validation_suite_missing_fingers():
    suite = AvatarValidationSuite()
    stitched = AnimationClip("id", "sid", "g", 1.0, "RIGHT", [
        AnimationClipKeyframe(0.0, {}, {}) # No fingers!
    ])
    
    res = suite.check_missing_finger_tracks(stitched)
    assert not res.passed
    assert "missing_bones_counts" in res.details

def test_avatar_validation_suite_broken_transitions():
    suite = AvatarValidationSuite()
    stitched = AnimationClip("id", "sid", "g", 1.0, "RIGHT", [
        AnimationClipKeyframe(0.0, {}, {"Wrist_R": [0.0, 0.0, 0.0]}),
        AnimationClipKeyframe(16.0, {}, {"Wrist_R": [10.0, 10.0, 10.0]}) # Big jump
    ])
    
    res = suite.check_broken_transitions(stitched)
    assert not res.passed
    assert res.details["max_spatial_jump"] > 10.0

def test_avatar_validation_suite_nmm_missing():
    suite = AvatarValidationSuite()
    ir_seq = [{"gloss": "HELP", "non_manual_markers": [{"tag": "furrowed"}]}]
    
    stitched = AnimationClip("id", "sid", "g", 1.0, "RIGHT", [
        AnimationClipKeyframe(0.0, {}, {}, blendshape_weights={}) # No blendshapes
    ])
    
    res = suite.check_nmm_timing_applied(ir_seq, stitched)
    assert not res.passed

def test_avatar_validation_suite_sequence_ordering():
    suite = AvatarValidationSuite()
    stitched = AnimationClip("id", "sid", "g", 1.0, "RIGHT", [
        AnimationClipKeyframe(16.0, {}, {}),
        AnimationClipKeyframe(0.0, {}, {}) # Out of order
    ])
    
    res = suite.check_sequence_ordering(stitched)
    assert not res.passed
