"""Facial and Non-Manual Marker (NMM) Controller (Prompt 76).

Connects ISL representation non-manual marker tags to avatar controllers:
- Blendshape scheduling with AHDR timing envelopes
- Head pose channels (nod pitch, headshake yaw, tilt roll, forward thrust)
- Body lean channels (forward/back spine flexion, lateral lean, role shift yaw)
- Eyebrow state tracking (FURROWED, RAISED, NEUTRAL)
- Keyframe timing synchronization (16.67 ms / 60 FPS)
"""

from dataclasses import dataclass, field, asdict
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from src.avatar.animation_clip_library import AnimationClipKeyframe
from src.avatar.motion_retargeter import VectorMath


@dataclass
class NonManualTagSpec:
    """Specification configuration for an active non-manual marker tag."""
    tag: str
    category: str = "general"
    start_ms: float = 0.0
    end_ms: float = 1000.0
    intensity: float = 1.0
    attack_ms: float = 100.0
    decay_ms: float = 80.0

    def get_envelope_weight(self, timestamp_ms: float) -> float:
        """Calculate AHDR envelope weight [0.0, 1.0] at given timestamp."""
        if timestamp_ms < self.start_ms or timestamp_ms > self.end_ms:
            return 0.0

        dur = max(1.0, self.end_ms - self.start_ms)
        rel_t = timestamp_ms - self.start_ms

        # Attack phase
        if rel_t < self.attack_ms:
            u = rel_t / max(1.0, self.attack_ms)
            return self.intensity * (u * u * (3.0 - 2.0 * u))

        # Decay phase
        time_rem = self.end_ms - timestamp_ms
        if time_rem < self.decay_ms:
            u = time_rem / max(1.0, self.decay_ms)
            return self.intensity * (u * u * (3.0 - 2.0 * u))

        # Peak hold phase
        return self.intensity


@dataclass
class NMMFrameState:
    """Evaluated Non-Manual Marker state snapshot for a single keyframe timestamp."""
    timestamp_ms: float
    active_tags: List[str] = field(default_factory=list)
    eyebrow_state: str = "NEUTRAL"  # "FURROWED", "RAISED", "NEUTRAL"
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # pitch, yaw, roll (radians)
    head_translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [x, y, z] meters
    body_lean_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # pitch, yaw, roll (radians)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "active_tags": self.active_tags,
            "eyebrow_state": self.eyebrow_state,
            "blendshapes": self.blendshapes,
            "head_pose_euler": self.head_pose_euler,
            "head_translation": self.head_translation,
            "body_lean_euler": self.body_lean_euler,
        }


class NMMController:
    """Controller driving facial blendshapes, head pose, body lean, and eyebrow states."""

    # 32 FACS / ISL Blendshapes dictionary matching Section 7 of AVATAR_SPEC.md
    BLENDSHAPE_KEYS = [
        "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "cheekPuff", "cheekSuck", "jawOpen", "mouthPucker", "mouthFunnel",
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthing_phoneme_01", "mouthing_phoneme_02", "mouthing_phoneme_03",
        "mouthing_phoneme_04", "mouthing_phoneme_05", "mouthing_phoneme_06",
        "mouthing_phoneme_07", "mouthing_phoneme_08", "mouthing_phoneme_09",
        "mouthing_phoneme_10", "mouthing_phoneme_11", "mouthing_phoneme_12",
        "mouthing_phoneme_13", "mouthing_phoneme_14", "mouthing_phoneme_15"
    ]

    def evaluate_nmm_state(
        self, tag_specs: List[NonManualTagSpec], timestamp_ms: float
    ) -> NMMFrameState:
        """Evaluate non-manual marker frame state at given timestamp."""
        active_tags: List[str] = []
        blendshapes: Dict[str, float] = {k: 0.0 for k in self.BLENDSHAPE_KEYS}
        
        head_pitch = 0.0
        head_yaw = 0.0
        head_roll = 0.0
        head_z = 0.0

        body_pitch = 0.0
        body_yaw = 0.0
        body_roll = 0.0

        has_furrowed = False
        has_raised = False

        for spec in tag_specs:
            w = spec.get_envelope_weight(timestamp_ms)
            if w > 0.01:
                active_tags.append(spec.tag)

                tag_lower = spec.tag.lower()

                # 1. Eyebrow State & Blendshapes
                if "furrowed" in tag_lower or "q_marker_wh" in tag_lower:
                    has_furrowed = True
                    blendshapes["browDownLeft"] = max(blendshapes["browDownLeft"], 0.85 * w)
                    blendshapes["browDownRight"] = max(blendshapes["browDownRight"], 0.85 * w)

                if "raised" in tag_lower or "q_marker_yn" in tag_lower:
                    has_raised = True
                    blendshapes["browOuterUpLeft"] = max(blendshapes["browOuterUpLeft"], 0.80 * w)
                    blendshapes["browOuterUpRight"] = max(blendshapes["browOuterUpRight"], 0.80 * w)

                # 2. Eye & Facial Blendshapes
                if "squint" in tag_lower:
                    blendshapes["eyeSquintLeft"] = max(blendshapes["eyeSquintLeft"], 0.75 * w)
                    blendshapes["eyeSquintRight"] = max(blendshapes["eyeSquintRight"], 0.75 * w)

                if "expressive" in tag_lower or "wide" in tag_lower:
                    blendshapes["eyeWideLeft"] = max(blendshapes["eyeWideLeft"], 0.80 * w)
                    blendshapes["eyeWideRight"] = max(blendshapes["eyeWideRight"], 0.80 * w)

                if "mouth_open" in tag_lower or "open_ah" in tag_lower:
                    blendshapes["jawOpen"] = max(blendshapes["jawOpen"], 0.60 * w)

                if "mouth_closed" in tag_lower or "pucker" in tag_lower:
                    blendshapes["mouthPucker"] = max(blendshapes["mouthPucker"], 0.50 * w)

                if "mouthing" in tag_lower:
                    blendshapes["mouthing_phoneme_01"] = max(blendshapes["mouthing_phoneme_01"], 0.70 * w)

                # 3. Head Pose Channels
                if "nod" in tag_lower:
                    # Pitch nod oscillation (freq ~ 3 Hz)
                    osc = math.sin(2.0 * math.pi * 0.003 * timestamp_ms)
                    head_pitch += math.radians(15.0) * osc * w

                if "shake" in tag_lower or "negation" in tag_lower:
                    # Yaw shake oscillation (freq ~ 4 Hz)
                    osc = math.sin(2.0 * math.pi * 0.004 * timestamp_ms)
                    head_yaw += math.radians(25.0) * osc * w

                if "tilt" in tag_lower:
                    head_roll += math.radians(15.0) * w

                if "thrust" in tag_lower or "focus" in tag_lower:
                    head_z += 0.04 * w

                # 4. Body Lean Channels
                if "body_lean_forward" in tag_lower:
                    body_pitch += math.radians(15.0) * w

                if "body_lean_back" in tag_lower:
                    body_pitch -= math.radians(10.0) * w

                if "role_shift" in tag_lower:
                    body_yaw += math.radians(20.0) * w

        # Determine eyebrow state
        if has_furrowed:
            eyebrow_state = "FURROWED"
        elif has_raised:
            eyebrow_state = "RAISED"
        else:
            eyebrow_state = "NEUTRAL"

        return NMMFrameState(
            timestamp_ms=timestamp_ms,
            active_tags=active_tags,
            eyebrow_state=eyebrow_state,
            blendshapes=blendshapes,
            head_pose_euler=[head_pitch, head_yaw, head_roll],
            head_translation=[0.0, 0.0, head_z],
            body_lean_euler=[body_pitch, body_yaw, body_roll]
        )

    def synchronize_nmm_to_keyframes(
        self, keyframes: List[AnimationClipKeyframe], tag_specs: List[NonManualTagSpec]
    ) -> List[AnimationClipKeyframe]:
        """Synchronize NMM blendshapes, head pose, and body lean to keyframes."""
        if not keyframes:
            return []

        synced_keyframes = []
        for kf in keyframes:
            nmm_state = self.evaluate_nmm_state(tag_specs, kf.timestamp_ms)

            # Update blendshapes
            new_blendshapes = dict(kf.blendshape_weights)
            for bs_key, w in nmm_state.blendshapes.items():
                if w > 0.0:
                    new_blendshapes[bs_key] = max(new_blendshapes.get(bs_key, 0.0), w)

            # Update Head bone rotation quaternion
            new_rotations = dict(kf.joint_rotations)
            if any(abs(a) > 1e-4 for a in nmm_state.head_pose_euler):
                base_head_q = kf.joint_rotations.get("Head", [0.0, 0.0, 0.0, 1.0])
                bp, by, br = VectorMath.quat_to_euler(base_head_q)
                
                hp = bp + nmm_state.head_pose_euler[0]
                hy = by + nmm_state.head_pose_euler[1]
                hr = br + nmm_state.head_pose_euler[2]

                new_rotations["Head"] = VectorMath.euler_to_quat(hp, hy, hr)

            # Update Spine / Chest body lean rotation quaternion
            if any(abs(a) > 1e-4 for a in nmm_state.body_lean_euler):
                base_spine_q = kf.joint_rotations.get("Spine_02", [0.0, 0.0, 0.0, 1.0])
                sp, sy, sr = VectorMath.quat_to_euler(base_spine_q)
                
                np_spine = sp + nmm_state.body_lean_euler[0]
                ny_spine = sy + nmm_state.body_lean_euler[1]
                nr_spine = sr + nmm_state.body_lean_euler[2]

                new_rotations["Spine_02"] = VectorMath.euler_to_quat(np_spine, ny_spine, nr_spine)

            synced_kf = AnimationClipKeyframe(
                timestamp_ms=kf.timestamp_ms,
                joint_rotations=new_rotations,
                joint_positions=dict(kf.joint_positions),
                blendshape_weights=new_blendshapes
            )
            synced_keyframes.append(synced_kf)

        return synced_keyframes
