"""Automated Animation Clip Validator Subsystem (Prompt 73).

Provides automated checks prioritizing correctness over visual beauty:
- Missing clips check
- Wrong IDs check
- Incorrect handedness metadata check
- Invalid animation duration check
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.avatar.animation_clip_library import AnimationClipLibrary, AnimationClip
from src.avatar.asset_registry import AvatarSignAssetRegistry


@dataclass
class ValidationError:
    """Error entry recorded during automated clip validation."""
    error_type: str  # "MISSING_CLIP", "WRONG_ID", "INCORRECT_HANDEDNESS", "INVALID_DURATION"
    target_identifier: str  # clip_id, sign_id, or gloss
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Summary report of automated animation clip validation execution."""
    total_clips_checked: int
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_clips_checked": self.total_clips_checked,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "errors": [
                {
                    "error_type": err.error_type,
                    "target_identifier": err.target_identifier,
                    "message": err.message,
                    "details": err.details,
                }
                for err in self.errors
            ],
            "warnings": self.warnings,
        }


class AnimationClipValidator:
    """Automated validator for 3D avatar animation clips."""

    @staticmethod
    def check_missing_clips(
        controlled_vocabulary: List[str],
        library: AnimationClipLibrary
    ) -> List[ValidationError]:
        """Check if any controlled vocabulary term is missing an animation clip."""
        errors = []
        for term in controlled_vocabulary:
            term_clean = term.strip()
            clip = library.get_clip_by_gloss(term_clean)
            if clip is None:
                errors.append(
                    ValidationError(
                        error_type="MISSING_CLIP",
                        target_identifier=term_clean,
                        message=f"Missing animation clip for controlled vocabulary term '{term_clean}'.",
                        details={"gloss": term_clean}
                    )
                )
        return errors

    @staticmethod
    def check_wrong_ids(
        registry: AvatarSignAssetRegistry,
        library: AnimationClipLibrary
    ) -> List[ValidationError]:
        """Check for clip IDs that do not match their corresponding asset registry entry."""
        errors = []
        for clip in library.list_clips():
            registry_asset = registry.get_asset(clip.sign_id)
            if registry_asset is None:
                errors.append(
                    ValidationError(
                        error_type="WRONG_ID",
                        target_identifier=clip.clip_id,
                        message=f"Clip sign_id '{clip.sign_id}' not found in sign asset registry.",
                        details={"clip_id": clip.clip_id, "sign_id": clip.sign_id}
                    )
                )
            else:
                if registry_asset.gloss.upper() != clip.gloss.upper():
                    errors.append(
                        ValidationError(
                            error_type="WRONG_ID",
                            target_identifier=clip.clip_id,
                            message=f"Gloss mismatch for clip '{clip.clip_id}': clip gloss '{clip.gloss}' vs registry gloss '{registry_asset.gloss}'.",
                            details={"clip_gloss": clip.gloss, "registry_gloss": registry_asset.gloss}
                        )
                    )
                if registry_asset.dominant_hand.upper() != clip.dominant_hand.upper():
                    errors.append(
                        ValidationError(
                            error_type="WRONG_ID",
                            target_identifier=clip.clip_id,
                            message=f"Handedness metadata mismatch with registry for clip '{clip.clip_id}': clip '{clip.dominant_hand}' vs registry '{registry_asset.dominant_hand}'.",
                            details={"clip_handedness": clip.dominant_hand, "registry_handedness": registry_asset.dominant_hand}
                        )
                    )
        return errors

    @staticmethod
    def check_handedness_metadata(library: AnimationClipLibrary) -> List[ValidationError]:
        """Inspect keyframe joint motions to verify declared dominant_hand metadata."""
        errors = []
        for clip in library.list_clips():
            if not clip.keyframes:
                continue

            right_arm_moved = False
            left_arm_moved = False

            first_kf = clip.keyframes[0]
            last_kf = clip.keyframes[-1]

            # Measure right arm motion
            if "Wrist_R" in first_kf.joint_positions and "Wrist_R" in last_kf.joint_positions:
                pos_f = first_kf.joint_positions["Wrist_R"]
                pos_l = last_kf.joint_positions["Wrist_R"]
                dist_r = sum((a - b) ** 2 for a, b in zip(pos_f, pos_l)) ** 0.5
                if dist_r > 0.01:
                    right_arm_moved = True

            # Measure left arm motion
            if "Wrist_L" in first_kf.joint_positions and "Wrist_L" in last_kf.joint_positions:
                pos_f = first_kf.joint_positions["Wrist_L"]
                pos_l = last_kf.joint_positions["Wrist_L"]
                dist_l = sum((a - b) ** 2 for a, b in zip(pos_f, pos_l)) ** 0.5
                if dist_l > 0.01:
                    left_arm_moved = True

            declared_hand = clip.dominant_hand.upper()

            if declared_hand == "RIGHT" and not right_arm_moved:
                errors.append(
                    ValidationError(
                        error_type="INCORRECT_HANDEDNESS",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' declared dominant_hand='RIGHT' but right arm exhibits no keyframe trajectory motion.",
                        details={"clip_id": clip.clip_id, "declared_hand": declared_hand, "right_moved": right_arm_moved}
                    )
                )

            if declared_hand == "LEFT" and not left_arm_moved:
                errors.append(
                    ValidationError(
                        error_type="INCORRECT_HANDEDNESS",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' declared dominant_hand='LEFT' but left arm exhibits no keyframe trajectory motion.",
                        details={"clip_id": clip.clip_id, "declared_hand": declared_hand, "left_moved": left_arm_moved}
                    )
                )

            if declared_hand == "BOTH" and not (right_arm_moved and left_arm_moved):
                errors.append(
                    ValidationError(
                        error_type="INCORRECT_HANDEDNESS",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' declared dominant_hand='BOTH' but one or both arms exhibit no motion (right_moved={right_arm_moved}, left_moved={left_arm_moved}).",
                        details={"clip_id": clip.clip_id, "right_moved": right_arm_moved, "left_moved": left_arm_moved}
                    )
                )

            if declared_hand == "RIGHT" and left_arm_moved:
                errors.append(
                    ValidationError(
                        error_type="INCORRECT_HANDEDNESS",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' declared dominant_hand='RIGHT' but left arm moves unexpectedly.",
                        details={"clip_id": clip.clip_id, "left_moved": left_arm_moved}
                    )
                )

            if declared_hand == "LEFT" and right_arm_moved:
                errors.append(
                    ValidationError(
                        error_type="INCORRECT_HANDEDNESS",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' declared dominant_hand='LEFT' but right arm moves unexpectedly.",
                        details={"clip_id": clip.clip_id, "right_moved": right_arm_moved}
                    )
                )

        return errors

    @staticmethod
    def check_animation_durations(library: AnimationClipLibrary) -> List[ValidationError]:
        """Validate animation clip durations and keyframe timestamp monotonic progression."""
        errors = []
        for clip in library.list_clips():
            if clip.duration <= 0:
                errors.append(
                    ValidationError(
                        error_type="INVALID_DURATION",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' has invalid non-positive duration: {clip.duration}.",
                        details={"duration": clip.duration}
                    )
                )

            if not clip.keyframes:
                errors.append(
                    ValidationError(
                        error_type="INVALID_DURATION",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' contains zero keyframes.",
                        details={"keyframe_count": 0}
                    )
                )
                continue

            prev_time = -1.0
            for idx, kf in enumerate(clip.keyframes):
                if kf.timestamp_ms < prev_time:
                    errors.append(
                        ValidationError(
                            error_type="INVALID_DURATION",
                            target_identifier=clip.clip_id,
                            message=f"Clip '{clip.clip_id}' keyframe at index {idx} has non-monotonic timestamp: {kf.timestamp_ms} < {prev_time}.",
                            details={"index": idx, "timestamp_ms": kf.timestamp_ms, "prev_time": prev_time}
                        )
                    )
                prev_time = kf.timestamp_ms

            # Check matching final timestamp vs declared duration (in ms)
            final_ts_ms = clip.keyframes[-1].timestamp_ms
            expected_ms = clip.duration * 1000.0
            if abs(final_ts_ms - expected_ms) > 50.0:  # Allow 50ms tolerance
                errors.append(
                    ValidationError(
                        error_type="INVALID_DURATION",
                        target_identifier=clip.clip_id,
                        message=f"Clip '{clip.clip_id}' final keyframe timestamp ({final_ts_ms:.1f}ms) deviates from declared duration ({expected_ms:.1f}ms).",
                        details={"final_ts_ms": final_ts_ms, "declared_duration_ms": expected_ms}
                    )
                )

        return errors

    @classmethod
    def run_full_validation(
        cls,
        controlled_vocabulary: List[str],
        registry: AvatarSignAssetRegistry,
        library: AnimationClipLibrary
    ) -> ValidationReport:
        """Run all automated checks and return aggregate ValidationReport."""
        all_errors = []
        all_errors.extend(cls.check_missing_clips(controlled_vocabulary, library))
        all_errors.extend(cls.check_wrong_ids(registry, library))
        all_errors.extend(cls.check_handedness_metadata(library))
        all_errors.extend(cls.check_animation_durations(library))

        is_valid = len(all_errors) == 0
        return ValidationReport(
            total_clips_checked=len(library.list_clips()),
            is_valid=is_valid,
            errors=all_errors,
            warnings=[]
        )
