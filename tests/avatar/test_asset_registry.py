"""Unit tests for AvatarSignAssetRegistry (Prompt 72)."""

import json
import pytest
from pathlib import Path
from src.avatar.asset_registry import (
    SignAssetMetadata,
    ValidationStatus,
    AvatarSignAssetRegistry,
)


def test_sign_asset_metadata_initialization():
    """Test valid initialization of SignAssetMetadata."""
    asset = SignAssetMetadata(
        sign_id="SIGN_DOCTOR_01",
        gloss="DOCTOR",
        animation_asset="assets/motions/doctor.anim",
        duration=1.25,
        dominant_hand="RIGHT",
        start_pose={"wrist": [0, 0, 0]},
        end_pose={"wrist": [1, 1, 1]},
        required_facial_markers=["browDownLeft"],
        validation_status=ValidationStatus.VALIDATED,
        reviewer="Linguist_A",
        version="1.0.0"
    )

    assert asset.sign_id == "SIGN_DOCTOR_01"
    assert asset.gloss == "DOCTOR"
    assert asset.duration == 1.25
    assert asset.dominant_hand == "RIGHT"
    assert asset.is_validated is True
    assert asset.reviewer == "Linguist_A"


def test_sign_asset_metadata_validation_errors():
    """Test validation errors for invalid SignAssetMetadata inputs."""
    with pytest.raises(ValueError, match="sign_id"):
        SignAssetMetadata(
            sign_id="", gloss="DOCTOR", animation_asset="path.anim", duration=1.0, dominant_hand="RIGHT"
        )

    with pytest.raises(ValueError, match="gloss"):
        SignAssetMetadata(
            sign_id="ID1", gloss="", animation_asset="path.anim", duration=1.0, dominant_hand="RIGHT"
        )

    with pytest.raises(ValueError, match="duration"):
        SignAssetMetadata(
            sign_id="ID1", gloss="GLOSS", animation_asset="path.anim", duration=-0.5, dominant_hand="RIGHT"
        )

    with pytest.raises(ValueError, match="dominant_hand"):
        SignAssetMetadata(
            sign_id="ID1", gloss="GLOSS", animation_asset="path.anim", duration=1.0, dominant_hand="FOOT"
        )


def test_registry_registration_and_retrieval():
    """Test registering and retrieving assets by ID and gloss."""
    registry = AvatarSignAssetRegistry()
    asset1 = SignAssetMetadata(
        sign_id="SIGN_HOSPITAL_01",
        gloss="HOSPITAL",
        animation_asset="assets/hospital.anim",
        duration=1.5,
        dominant_hand="RIGHT",
        validation_status=ValidationStatus.UNVALIDATED
    )

    registry.register_asset(asset1)
    retrieved = registry.get_asset("SIGN_HOSPITAL_01")
    assert retrieved is asset1

    by_gloss = registry.get_asset_by_gloss("hospital")
    assert by_gloss is asset1

    assert registry.get_asset("NON_EXISTENT") is None
    assert registry.get_asset_by_gloss("NON_EXISTENT") is None


def test_unvalidated_assets_marking_and_filtering():
    """Test that unvalidated assets are clearly marked and filterable."""
    registry = AvatarSignAssetRegistry.create_seeded_registry()
    unvalidated = registry.get_unvalidated_assets()
    validated = registry.get_validated_assets()

    assert len(unvalidated) > 0
    assert len(validated) > 0

    for asset in unvalidated:
        assert asset.validation_status == ValidationStatus.UNVALIDATED
        assert asset.is_validated is False

    for asset in validated:
        assert asset.validation_status == ValidationStatus.VALIDATED
        assert asset.is_validated is True
        assert asset.reviewer is not None


def test_validate_asset_workflow():
    """Test transitioning an asset from unvalidated to validated."""
    registry = AvatarSignAssetRegistry.create_seeded_registry()
    unvalidated_assets = registry.get_unvalidated_assets()
    target_asset = unvalidated_assets[0]

    updated = registry.validate_asset(
        sign_id=target_asset.sign_id, reviewer="Expert_Reviewer_42", version="1.1.0"
    )

    assert updated.validation_status == ValidationStatus.VALIDATED
    assert updated.is_validated is True
    assert updated.reviewer == "Expert_Reviewer_42"
    assert updated.version == "1.1.0"

    # Test error handling when validating non-existent sign
    with pytest.raises(KeyError):
        registry.validate_asset("MISSING_SIGN", reviewer="Reviewer")

    with pytest.raises(ValueError):
        registry.validate_asset(target_asset.sign_id, reviewer="")


def test_unvalidate_asset_workflow():
    """Test unvalidating an asset."""
    registry = AvatarSignAssetRegistry.create_seeded_registry()
    validated_assets = registry.get_validated_assets()
    target = validated_assets[0]

    unvalidated = registry.unvalidate_asset(target.sign_id)
    assert unvalidated.validation_status == ValidationStatus.UNVALIDATED
    assert unvalidated.is_validated is False


def test_json_serialization(tmp_path: Path):
    """Test exporting and saving/loading registry JSON."""
    registry = AvatarSignAssetRegistry.create_seeded_registry()
    json_file = tmp_path / "registry.json"

    registry.save_json(json_file)
    assert json_file.exists()

    new_registry = AvatarSignAssetRegistry()
    new_registry.load_json(json_file)

    assert len(new_registry.list_assets()) == len(registry.list_assets())
    doctor = new_registry.get_asset("SIGN_DOCTOR_01")
    assert doctor is not None
    assert doctor.gloss == "DOCTOR"
    assert doctor.validation_status == ValidationStatus.VALIDATED
    assert doctor.reviewer == "DHH_Linguist_01"
