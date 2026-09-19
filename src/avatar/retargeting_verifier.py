"""Retargeting Verifier Subsystem (Prompt 74).

Automated 5-point verification suite evaluating retargeted motion for:
1. Finger alignment
2. Wrist orientation
3. Joint constraints
4. Body scale
5. Handedness
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, List, Optional, Tuple
from src.avatar.animation_clip_library import AnimationClip, AnimationClipKeyframe
from src.avatar.motion_retargeter import HumanoidRigConfig, VectorMath, JointLimit


@dataclass
class VerificationError:
    """Individual verification violation log."""
    criterion: str  # "FINGER_ALIGNMENT", "WRIST_ORIENTATION", "JOINT_CONSTRAINTS", "BODY_SCALE", "HANDEDNESS"
    keyframe_index: int
    timestamp_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetargetingVerificationReport:
    """Summary report for 5-point retargeting verification execution."""
    clip_id: str
    is_valid: bool
    finger_alignment_valid: bool
    wrist_orientation_valid: bool
    joint_constraints_valid: bool
    body_scale_valid: bool
    handedness_valid: bool
    errors: List[VerificationError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "is_valid": self.is_valid,
            "finger_alignment_valid": self.finger_alignment_valid,
            "wrist_orientation_valid": self.wrist_orientation_valid,
            "joint_constraints_valid": self.joint_constraints_valid,
            "body_scale_valid": self.body_scale_valid,
            "handedness_valid": self.handedness_valid,
            "error_count": len(self.errors),
            "errors": [
                {
                    "criterion": err.criterion,
                    "keyframe_index": err.keyframe_index,
                    "timestamp_ms": err.timestamp_ms,
                    "message": err.message,
                    "details": err.details,
                }
                for err in self.errors
            ],
            "metrics": self.metrics,
        }


class RetargetingVerifier:
    """5-Point Verifier for Unity Humanoid Rig retargeted motion."""

    def __init__(self, target_config: Optional[HumanoidRigConfig] = None) -> None:
        self.target_config = target_config or HumanoidRigConfig()

    def verify_finger_alignment(
        self, clip: AnimationClip
    ) -> Tuple[bool, List[VerificationError], Dict[str, Any]]:
        """Verify finger joint angle alignment and anatomical limits."""
        errors = []
        total_finger_joints_checked = 0

        finger_prefixes = ["Thumb_", "Index_", "Middle_", "Ring_", "Pinky_"]

        for idx, kf in enumerate(clip.keyframes):
            for bone, rot in kf.joint_rotations.items():
                if any(bone.startswith(pref) for pref in finger_prefixes):
                    total_finger_joints_checked += 1
                    pitch, yaw, roll = VectorMath.quat_to_euler(rot)
                    
                    if bone in self.target_config.joint_limits:
                        limit = self.target_config.joint_limits[bone]
                        if not (limit.min_pitch - 1e-3 <= pitch <= limit.max_pitch + 1e-3 and
                                limit.min_yaw - 1e-3 <= yaw <= limit.max_yaw + 1e-3):
                            errors.append(
                                VerificationError(
                                    criterion="FINGER_ALIGNMENT",
                                    keyframe_index=idx,
                                    timestamp_ms=kf.timestamp_ms,
                                    message=f"Finger joint '{bone}' exceeds anatomical alignment bounds.",
                                    details={"bone": bone, "pitch": pitch, "yaw": yaw, "roll": roll}
                                )
                            )

        is_valid = len(errors) == 0
        metrics = {
            "finger_joints_checked": total_finger_joints_checked,
            "finger_alignment_errors": len(errors)
        }
        return is_valid, errors, metrics

    def verify_wrist_orientation(
        self, clip: AnimationClip
    ) -> Tuple[bool, List[VerificationError], Dict[str, Any]]:
        """Verify wrist rotation matrix orthogonality and roll/pitch/yaw bounds."""
        errors = []
        wrists_checked = 0

        for idx, kf in enumerate(clip.keyframes):
            for wrist_bone in ["Wrist_L", "Wrist_R"]:
                if wrist_bone in kf.joint_rotations:
                    wrists_checked += 1
                    q = kf.joint_rotations[wrist_bone]
                    
                    norm = VectorMath.norm(q) if len(q) == 4 else 1.0
                    if abs(norm - 1.0) > 1e-3:
                        errors.append(
                            VerificationError(
                                criterion="WRIST_ORIENTATION",
                                keyframe_index=idx,
                                timestamp_ms=kf.timestamp_ms,
                                message=f"Wrist joint '{wrist_bone}' quaternion is not normalized (norm={norm:.5f}).",
                                details={"bone": wrist_bone, "norm": norm}
                            )
                        )

                    pitch, yaw, roll = VectorMath.quat_to_euler(q)
                    if wrist_bone in self.target_config.joint_limits:
                        limit = self.target_config.joint_limits[wrist_bone]
                        if not (limit.min_pitch - 1e-3 <= pitch <= limit.max_pitch + 1e-3 and
                                limit.min_yaw - 1e-3 <= yaw <= limit.max_yaw + 1e-3 and
                                limit.min_roll - 1e-3 <= roll <= limit.max_roll + 1e-3):
                            errors.append(
                                VerificationError(
                                    criterion="WRIST_ORIENTATION",
                                    keyframe_index=idx,
                                    timestamp_ms=kf.timestamp_ms,
                                    message=f"Wrist joint '{wrist_bone}' pitch/yaw/roll out of anatomical bounds.",
                                    details={"bone": wrist_bone, "pitch": pitch, "yaw": yaw, "roll": roll}
                                )
                            )

        is_valid = len(errors) == 0
        metrics = {
            "wrists_checked": wrists_checked,
            "wrist_orientation_errors": len(errors)
        }
        return is_valid, errors, metrics

    def verify_joint_constraints(
        self, clip: AnimationClip
    ) -> Tuple[bool, List[VerificationError], Dict[str, Any]]:
        """Verify clamping of all 67 humanoid rig joints to anatomical limits."""
        errors = []
        total_joints_checked = 0

        for idx, kf in enumerate(clip.keyframes):
            for bone, rot in kf.joint_rotations.items():
                total_joints_checked += 1
                if bone in self.target_config.joint_limits:
                    limit = self.target_config.joint_limits[bone]
                    pitch, yaw, roll = VectorMath.quat_to_euler(rot)
                    
                    if (pitch < limit.min_pitch - 1e-3 or pitch > limit.max_pitch + 1e-3 or
                        yaw < limit.min_yaw - 1e-3 or yaw > limit.max_yaw + 1e-3 or
                        roll < limit.min_roll - 1e-3 or roll > limit.max_roll + 1e-3):
                        errors.append(
                            VerificationError(
                                criterion="JOINT_CONSTRAINTS",
                                keyframe_index=idx,
                                timestamp_ms=kf.timestamp_ms,
                                message=f"Joint '{bone}' violates rotation constraint bounds.",
                                details={"bone": bone, "pitch": pitch, "yaw": yaw, "roll": roll}
                            )
                        )

        is_valid = len(errors) == 0
        metrics = {
            "total_joints_checked": total_joints_checked,
            "constraint_violations": len(errors)
        }
        return is_valid, errors, metrics

    def verify_body_scale(
        self, source_clip: AnimationClip, retargeted_clip: AnimationClip, scale_factor: float
    ) -> Tuple[bool, List[VerificationError], Dict[str, Any]]:
        """Verify spatial displacement position scaling accuracy."""
        errors = []
        max_dev = 0.0
        positions_checked = 0

        side_map = {}
        for bone in self.target_config.bones:
            if bone.endswith("_R"):
                side_map[bone] = bone[:-2] + "_L"
            elif bone.endswith("_L"):
                side_map[bone] = bone[:-2] + "_R"

        is_mirrored = (source_clip.dominant_hand.upper() != retargeted_clip.dominant_hand.upper())

        for idx, (src_kf, ret_kf) in enumerate(zip(source_clip.keyframes, retargeted_clip.keyframes)):
            for bone, src_pos in src_kf.joint_positions.items():
                target_bone = side_map.get(bone, bone) if is_mirrored else bone
                if target_bone in ret_kf.joint_positions:
                    positions_checked += 1
                    ret_pos = ret_kf.joint_positions[target_bone]
                    x_mult = -1.0 if (is_mirrored and target_bone != bone) else 1.0
                    expected_pos = [src_pos[0] * scale_factor * x_mult, src_pos[1] * scale_factor, src_pos[2] * scale_factor]
                    
                    dev = math.sqrt(sum((r - e) ** 2 for r, e in zip(ret_pos, expected_pos)))
                    if dev > max_dev:
                        max_dev = dev

                    if dev > 1e-3:
                        errors.append(
                            VerificationError(
                                criterion="BODY_SCALE",
                                keyframe_index=idx,
                                timestamp_ms=ret_kf.timestamp_ms,
                                message=f"Bone '{target_bone}' retargeted position deviates from expected body scale factor ({scale_factor}).",
                                details={"bone": target_bone, "ret_pos": ret_pos, "expected_pos": expected_pos, "deviation": dev}
                            )
                        )

        is_valid = len(errors) == 0
        metrics = {
            "positions_checked": positions_checked,
            "max_scale_deviation": max_dev,
            "scale_errors": len(errors)
        }
        return is_valid, errors, metrics

    def verify_handedness(
        self, retargeted_clip: AnimationClip, target_handedness: Optional[str] = None
    ) -> Tuple[bool, List[VerificationError], Dict[str, Any]]:
        """Verify dominant hand trajectory allocation and handedness mirroring consistency."""
        errors = []
        declared_hand = (target_handedness or retargeted_clip.dominant_hand).upper()

        if retargeted_clip.keyframes:
            first_kf = retargeted_clip.keyframes[0]
            last_kf = retargeted_clip.keyframes[-1]

            right_moved = False
            left_moved = False

            if "Wrist_R" in first_kf.joint_positions and "Wrist_R" in last_kf.joint_positions:
                dist_r = math.sqrt(sum((a - b) ** 2 for a, b in zip(first_kf.joint_positions["Wrist_R"], last_kf.joint_positions["Wrist_R"])))
                if dist_r > 0.01:
                    right_moved = True

            if "Wrist_L" in first_kf.joint_positions and "Wrist_L" in last_kf.joint_positions:
                dist_l = math.sqrt(sum((a - b) ** 2 for a, b in zip(first_kf.joint_positions["Wrist_L"], last_kf.joint_positions["Wrist_L"])))
                if dist_l > 0.01:
                    left_moved = True

            if declared_hand == "RIGHT" and not right_moved:
                errors.append(
                    VerificationError(
                        criterion="HANDEDNESS",
                        keyframe_index=0,
                        timestamp_ms=0.0,
                        message="Declared dominant_hand='RIGHT' but right arm exhibits no keyframe trajectory motion.",
                        details={"declared_hand": declared_hand, "right_moved": right_moved}
                    )
                )

            if declared_hand == "LEFT" and not left_moved:
                errors.append(
                    VerificationError(
                        criterion="HANDEDNESS",
                        keyframe_index=0,
                        timestamp_ms=0.0,
                        message="Declared dominant_hand='LEFT' but left arm exhibits no keyframe trajectory motion.",
                        details={"declared_hand": declared_hand, "left_moved": left_moved}
                    )
                )

        is_valid = len(errors) == 0
        metrics = {
            "declared_handedness": declared_hand,
            "handedness_errors": len(errors)
        }
        return is_valid, errors, metrics

    def verify_all(
        self,
        source_clip: AnimationClip,
        retargeted_clip: AnimationClip,
        scale_factor: float = 1.0,
        target_handedness: Optional[str] = None
    ) -> RetargetingVerificationReport:
        """Run all 5 verification routines and return aggregated RetargetingVerificationReport."""
        all_errors = []

        f_valid, f_errors, f_metrics = self.verify_finger_alignment(retargeted_clip)
        w_valid, w_errors, w_metrics = self.verify_wrist_orientation(retargeted_clip)
        j_valid, j_errors, j_metrics = self.verify_joint_constraints(retargeted_clip)
        s_valid, s_errors, s_metrics = self.verify_body_scale(source_clip, retargeted_clip, scale_factor)
        h_valid, h_errors, h_metrics = self.verify_handedness(retargeted_clip, target_handedness)

        all_errors.extend(f_errors)
        all_errors.extend(w_errors)
        all_errors.extend(j_errors)
        all_errors.extend(s_errors)
        all_errors.extend(h_errors)

        is_valid = f_valid and w_valid and j_valid and s_valid and h_valid

        combined_metrics = {
            **f_metrics,
            **w_metrics,
            **j_metrics,
            **s_metrics,
            **h_metrics,
        }

        return RetargetingVerificationReport(
            clip_id=retargeted_clip.clip_id,
            is_valid=is_valid,
            finger_alignment_valid=f_valid,
            wrist_orientation_valid=w_valid,
            joint_constraints_valid=j_valid,
            body_scale_valid=s_valid,
            handedness_valid=h_valid,
            errors=all_errors,
            metrics=combined_metrics
        )
