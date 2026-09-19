"""Avatar Validation Suite (Prompt 78).

Automated and manual validation suite for Avatar sequences.
Checks:
- All sign IDs resolve
- Sequence ordering preserved
- NMM timing applied
- No impossible rotations
- No missing finger tracks
- No broken transitions
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from src.avatar.animation_clip_library import AnimationClip, AnimationClipLibrary
from src.avatar.retargeting_verifier import RetargetingVerifier
from src.avatar.motion_retargeter import HumanoidRigConfig

@dataclass
class ValidationReportEntry:
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SuiteValidationReport:
    sequence_id: str
    is_valid: bool
    entries: List[ValidationReportEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "is_valid": self.is_valid,
            "checks": [
                {
                    "check_name": e.check_name,
                    "passed": e.passed,
                    "message": e.message,
                    "details": e.details
                } for e in self.entries
            ]
        }


class AvatarValidationSuite:
    """Validation suite for IR sequences and stitched AnimationClips."""

    def __init__(self, clip_library: Optional[AnimationClipLibrary] = None):
        self.clip_library = clip_library or AnimationClipLibrary.create_seeded_library()
        self.retargeting_verifier = RetargetingVerifier(HumanoidRigConfig())

    def check_sign_ids_resolve(self, ir_sequence: List[Dict[str, Any]]) -> ValidationReportEntry:
        """Verify all glosses map to valid clips in the library."""
        missing = []
        for item in ir_sequence:
            gloss = item.get("gloss", "HELP")
            if self.clip_library.get_clip_by_gloss(gloss) is None:
                missing.append(gloss)
        
        passed = len(missing) == 0
        msg = "All sign IDs resolved successfully." if passed else f"Missing clips for glosses: {missing}"
        return ValidationReportEntry("SIGN_IDS_RESOLVE", passed, msg, {"missing": missing})

    def check_sequence_ordering(self, clip: AnimationClip) -> ValidationReportEntry:
        """Ensure monotonic timestamps across the clip's keyframes."""
        if not clip.keyframes:
            return ValidationReportEntry("SEQUENCE_ORDERING", False, "Clip has no keyframes.")
            
        prev_time = -1.0
        violations = []
        for idx, kf in enumerate(clip.keyframes):
            if kf.timestamp_ms < prev_time:
                violations.append({"index": idx, "time": kf.timestamp_ms, "prev_time": prev_time})
            prev_time = kf.timestamp_ms

        passed = len(violations) == 0
        msg = "Sequence ordering preserved monotonically." if passed else f"Found {len(violations)} ordering violations."
        return ValidationReportEntry("SEQUENCE_ORDERING", passed, msg, {"violations": violations})

    def check_nmm_timing_applied(self, ir_sequence: List[Dict[str, Any]], clip: AnimationClip) -> ValidationReportEntry:
        """Verify NMM blendshape modulation was applied to keyframes when expected."""
        nmm_present = False
        for item in ir_sequence:
            if "non_manual_markers" in item and len(item["non_manual_markers"]) > 0:
                nmm_present = True
                break
                
        if not nmm_present:
            return ValidationReportEntry("NMM_TIMING", True, "No NMM tags present in IR sequence to check.", {})
            
        if not clip.keyframes:
            return ValidationReportEntry("NMM_TIMING", False, "Clip has no keyframes.", {})
            
        max_nmm_weight = 0.0
        nmm_keys = ["browDownLeft", "browOuterUpLeft", "eyeSquintLeft", "jawOpen", "mouthPucker"]
        
        for kf in clip.keyframes:
            for key in nmm_keys:
                w = kf.blendshape_weights.get(key, 0.0)
                if w > max_nmm_weight:
                    max_nmm_weight = w
                    
        passed = max_nmm_weight > 0.01
        msg = "NMM timing and blendshapes observed." if passed else "No NMM blendshapes detected despite tags in IR."
        return ValidationReportEntry("NMM_TIMING", passed, msg, {"max_nmm_weight": max_nmm_weight})

    def check_impossible_rotations(self, clip: AnimationClip) -> ValidationReportEntry:
        """Leverage RetargetingVerifier to check joint constraints."""
        is_valid, errors, metrics = self.retargeting_verifier.verify_joint_constraints(clip)
        msg = "No impossible rotations found." if is_valid else f"Found {len(errors)} constraint violations."
        return ValidationReportEntry(
            "IMPOSSIBLE_ROTATIONS", 
            is_valid, 
            msg, 
            {"error_count": len(errors), "metrics": metrics}
        )

    def check_missing_finger_tracks(self, clip: AnimationClip) -> ValidationReportEntry:
        """Ensure all 30 finger joints exist in every keyframe."""
        prefixes = ["Thumb_", "Index_", "Middle_", "Ring_", "Pinky_"]
        suffixes = ["01", "02", "03"]
        sides = ["L", "R"]
        
        required_fingers = [f"{p}{s}_{side}" for p in prefixes for s in suffixes for side in sides]
        
        if not clip.keyframes:
            return ValidationReportEntry("MISSING_FINGER_TRACKS", False, "Clip has no keyframes.", {})
            
        missing_counts = {bone: 0 for bone in required_fingers}
        
        for kf in clip.keyframes:
            for req_bone in required_fingers:
                if req_bone not in kf.joint_rotations:
                    missing_counts[req_bone] += 1
                    
        actual_missing = {k: v for k, v in missing_counts.items() if v > 0}
        passed = len(actual_missing) == 0
        
        msg = "All finger tracks present." if passed else f"Missing finger tracks in some frames."
        return ValidationReportEntry("MISSING_FINGER_TRACKS", passed, msg, {"missing_bones_counts": actual_missing})

    def check_broken_transitions(self, clip: AnimationClip) -> ValidationReportEntry:
        """Basic check to ensure keyframes don't abruptly jump between frames, indicating broken transitions."""
        if len(clip.keyframes) < 2:
            return ValidationReportEntry("BROKEN_TRANSITIONS", True, "Insufficient keyframes for transition check.")
            
        max_jump = 0.0
        jump_idx = -1
        
        for i in range(1, len(clip.keyframes)):
            kf1 = clip.keyframes[i-1]
            kf2 = clip.keyframes[i]
            
            if "Wrist_R" in kf1.joint_positions and "Wrist_R" in kf2.joint_positions:
                p1 = kf1.joint_positions["Wrist_R"]
                p2 = kf2.joint_positions["Wrist_R"]
                dist = sum((a - b)**2 for a, b in zip(p1, p2))**0.5
                if dist > max_jump:
                    max_jump = dist
                    jump_idx = i
                    
        passed = max_jump < 0.3
        msg = "No broken abrupt transitions detected." if passed else f"Abrupt spatial jump of {max_jump:.3f}m detected at kf {jump_idx}."
        return ValidationReportEntry("BROKEN_TRANSITIONS", passed, msg, {"max_spatial_jump": max_jump, "jump_kf_index": jump_idx})

    def validate_sequence(self, ir_sequence: List[Dict[str, Any]], stitched_clip: AnimationClip) -> SuiteValidationReport:
        """Run all validation checks on an IR sequence and its corresponding stitched output clip."""
        entries = [
            self.check_sign_ids_resolve(ir_sequence),
            self.check_sequence_ordering(stitched_clip),
            self.check_nmm_timing_applied(ir_sequence, stitched_clip),
            self.check_impossible_rotations(stitched_clip),
            self.check_missing_finger_tracks(stitched_clip),
            self.check_broken_transitions(stitched_clip)
        ]
        
        is_valid = all(e.passed for e in entries)
        
        return SuiteValidationReport(
            sequence_id=stitched_clip.clip_id,
            is_valid=is_valid,
            entries=entries
        )

    def export_validation_report(self, report: SuiteValidationReport, output_path: Union[str, Path]) -> None:
        """Serialize report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
