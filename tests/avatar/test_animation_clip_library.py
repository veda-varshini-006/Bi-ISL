"""Unit tests for AnimationClipLibrary and AnimationClipValidator (Prompt 73)."""

import pytest
from pathlib import Path
from src.avatar.animation_clip_library import (
    AnimationClipKeyframe,
    AnimationClip,
    AnimationClipLibrary,
)
from src.avatar.animation_clip_validator import (
    AnimationClipValidator,
    ValidationError,
    ValidationReport,
)
from src.avatar.asset_registry import AvatarSignAssetRegistry, SignAssetMetadata, ValidationStatus


def test_animation_clip_initialization():
    """Test valid creation of AnimationClip containers."""
    kf1 = AnimationClipKeyframe(
        timestamp_ms=0.0,
        joint_positions={"Wrist_R": [0.1, 1.2, 0.4]},
        joint_rotations={"UpperArm_R": [0.0, 0.0, 0.0, 1.0]}
    )
    kf2 = AnimationClipKeyframe(
        timestamp_ms=1000.0,
        joint_positions={"Wrist_R": [0.2, 1.2, 0.4]},
        joint_rotations={"UpperArm_R": [0.1, 0.1, 0.0, 1.0]}
    )

    clip = AnimationClip(
        clip_id="CLIP_DOCTOR_01",
        sign_id="SIGN_DOCTOR_01",
        gloss="DOCTOR",
        duration=1.0,
        dominant_hand="RIGHT",
        keyframes=[kf1, kf2]
    )

    assert clip.clip_id == "CLIP_DOCTOR_01"
    assert clip.sign_id == "SIGN_DOCTOR_01"
    assert clip.gloss == "DOCTOR"
    assert clip.duration == 1.0
    assert clip.dominant_hand == "RIGHT"
    assert len(clip.keyframes) == 2


def test_animation_clip_invalid_initialization():
    """Test ValueError triggers on invalid AnimationClip properties."""
    with pytest.raises(ValueError, match="clip_id"):
        AnimationClip("", "SIGN_01", "GLOSS", 1.0, "RIGHT")

    with pytest.raises(ValueError, match="duration"):
        AnimationClip("CLIP_01", "SIGN_01", "GLOSS", -0.5, "RIGHT")

    with pytest.raises(ValueError, match="dominant_hand"):
        AnimationClip("CLIP_01", "SIGN_01", "GLOSS", 1.0, "INVALID")


def test_seeded_library_loading():
    """Test pre-populated seeded clip library."""
    library = AnimationClipLibrary.create_seeded_library()
    clips = library.list_clips()

    assert len(clips) > 0
    doctor_clip = library.get_clip_by_gloss("DOCTOR")
    assert doctor_clip is not None
    assert doctor_clip.sign_id == "SIGN_DOCTOR_01"
    assert len(doctor_clip.keyframes) > 0


def test_check_missing_clips():
    """Test automated check for missing clips."""
    library = AnimationClipLibrary()
    vocabulary = ["DOCTOR", "HOSPITAL", "FEVER"]

    # Library has no clips loaded yet
    errors = AnimationClipValidator.check_missing_clips(vocabulary, library)
    assert len(errors) == 3
    assert all(e.error_type == "MISSING_CLIP" for e in errors)

    # Add DOCTOR clip
    library.add_clip(
        AnimationClip("CLIP_DOC", "SIGN_DOC", "DOCTOR", 1.0, "RIGHT")
    )
    errors_after = AnimationClipValidator.check_missing_clips(vocabulary, library)
    assert len(errors_after) == 2


def test_check_wrong_ids():
    """Test automated check for clip vs asset registry ID and metadata mismatches."""
    registry = AvatarSignAssetRegistry()
    registry.register_asset(
        SignAssetMetadata(
            sign_id="SIGN_DOCTOR_01",
            gloss="DOCTOR",
            animation_asset="doc.anim",
            duration=1.25,
            dominant_hand="RIGHT",
            validation_status=ValidationStatus.VALIDATED
        )
    )

    library = AnimationClipLibrary()
    # Mismatched sign_id not in registry
    library.add_clip(
        AnimationClip("CLIP_MISMATCH", "SIGN_UNKNOWN", "DOCTOR", 1.25, "RIGHT")
    )

    errors = AnimationClipValidator.check_wrong_ids(registry, library)
    assert len(errors) == 1
    assert errors[0].error_type == "WRONG_ID"
    assert "not found in sign asset registry" in errors[0].message


def test_check_incorrect_handedness_metadata():
    """Test automated check for handedness trajectory mismatches."""
    library = AnimationClipLibrary()

    # Clip declares RIGHT but right arm does not move, left arm moves
    kf1 = AnimationClipKeyframe(
        timestamp_ms=0.0,
        joint_positions={"Wrist_R": [0.1, 1.0, 0.3], "Wrist_L": [-0.1, 1.0, 0.3]}
    )
    kf2 = AnimationClipKeyframe(
        timestamp_ms=1000.0,
        joint_positions={"Wrist_R": [0.1, 1.0, 0.3], "Wrist_L": [-0.3, 1.2, 0.5]}  # Left moved
    )
    bad_clip = AnimationClip(
        clip_id="CLIP_BAD_HAND",
        sign_id="SIGN_DOCTOR_01",
        gloss="DOCTOR",
        duration=1.0,
        dominant_hand="RIGHT",
        keyframes=[kf1, kf2]
    )
    library.add_clip(bad_clip)

    errors = AnimationClipValidator.check_handedness_metadata(library)
    assert len(errors) >= 1
    assert any(e.error_type == "INCORRECT_HANDEDNESS" for e in errors)


def test_check_invalid_animation_duration():
    """Test automated check for duration and timestamp anomalies."""
    library = AnimationClipLibrary()

    # Non-monotonic keyframe timestamps
    kf1 = AnimationClipKeyframe(timestamp_ms=0.0)
    kf2 = AnimationClipKeyframe(timestamp_ms=800.0)
    kf3 = AnimationClipKeyframe(timestamp_ms=500.0)  # Non-monotonic jump back

    bad_dur_clip = AnimationClip(
        clip_id="CLIP_BAD_DUR",
        sign_id="SIGN_PAIN_01",
        gloss="PAIN",
        duration=1.0,  # Expected 1000ms, final is 500ms
        dominant_hand="RIGHT",
        keyframes=[kf1, kf2, kf3]
    )
    library.add_clip(bad_dur_clip)

    errors = AnimationClipValidator.check_animation_durations(library)
    assert len(errors) >= 1
    assert any(e.error_type == "INVALID_DURATION" for e in errors)


def test_seeded_library_full_validation():
    """Test that the seeded initial library passes full automated validation against controlled vocabulary."""
    library = AnimationClipLibrary.create_seeded_library()
    registry = AvatarSignAssetRegistry()

    # Pre-load registry matching seeded clips
    for clip in library.list_clips():
        registry.register_asset(
            SignAssetMetadata(
                sign_id=clip.sign_id,
                gloss=clip.gloss,
                animation_asset=f"assets/motions/{clip.gloss.lower()}.anim",
                duration=clip.duration,
                dominant_hand=clip.dominant_hand,
                validation_status=ValidationStatus.VALIDATED,
                reviewer="Linguist_01"
            )
        )

    controlled_vocab = [clip.gloss for clip in library.list_clips()]
    report = AnimationClipValidator.run_full_validation(controlled_vocab, registry, library)

    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_clips_checked == len(controlled_vocab)


def test_json_persistence(tmp_path: Path):
    """Test saving and loading animation clip library JSON."""
    library = AnimationClipLibrary.create_seeded_library()
    json_path = tmp_path / "clip_library.json"

    library.save_json(json_path)
    assert json_path.exists()

    loaded_lib = AnimationClipLibrary()
    loaded_lib.load_json(json_path)

    assert len(loaded_lib.list_clips()) == len(library.list_clips())
    doctor = loaded_lib.get_clip_by_gloss("DOCTOR")
    assert doctor is not None
    assert doctor.clip_id == "CLIP_DOCTOR_01"
    assert len(doctor.keyframes) > 0
