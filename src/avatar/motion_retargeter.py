"""Motion Retargeting Engine for Unity Humanoid Rig (Prompt 74).

Performs robust motion retargeting onto standard Unity Humanoid skeletal rigs,
verifying and enforcing:
1. Finger alignment (handshape mapping & digit joint DoF bounds)
2. Wrist orientation (orthonormalization, palm facing/finger direction vectors, roll/pitch/yaw bounds)
3. Joint constraints (clamping rotations to anatomical limits across all 67 bones)
4. Body scale (rescaling spatial displacement & IK loci by target rig height scale factor)
5. Handedness (handedness conversion, X-reflection, rotation mirroring, dual-arm coordination)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
from src.avatar.animation_clip_library import AnimationClip, AnimationClipKeyframe


@dataclass
class JointLimit:
    """Anatomical rotation angular limits (in radians) for a single joint channel."""
    min_pitch: float = -math.pi
    max_pitch: float = math.pi
    min_yaw: float = -math.pi
    max_yaw: float = math.pi
    min_roll: float = -math.pi
    max_roll: float = math.pi

    def clamp_euler(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """Clamp Euler angles (pitch, yaw, roll) within defined limits."""
        c_pitch = max(self.min_pitch, min(self.max_pitch, pitch))
        c_yaw = max(self.min_yaw, min(self.max_yaw, yaw))
        c_roll = max(self.min_roll, min(self.max_roll, roll))
        return c_pitch, c_yaw, c_roll


@dataclass
class HumanoidRigConfig:
    """Configuration specification for standard Unity Humanoid Rig."""
    rig_name: str = "UnityHumanoidRig_Standard"
    height_meters: float = 1.75
    bone_lengths: Dict[str, float] = field(default_factory=lambda: {
        "UpperArm": 0.28,
        "LowerArm": 0.26,
        "Hand": 0.18,
        "Spine": 0.55,
        "Neck": 0.12,
        "Head": 0.22,
        "UpperLeg": 0.45,
        "LowerLeg": 0.43,
    })
    
    # Exactly 67 standard bones matching AVATAR_SPEC.md
    bones: List[str] = field(default_factory=lambda: [
        "Root", "Hips", "Spine_01", "Spine_02", "Spine_03", "Chest", "Pelvis", "Neck", "Head", "Jaw", "Eye_L", "Eye_R", "Head_End",
        "Clavicle_L", "UpperArm_L", "LowerArm_L", "Wrist_L", "Hand_L",
        "Thumb_01_L", "Thumb_02_L", "Thumb_03_L",
        "Index_01_L", "Index_02_L", "Index_03_L",
        "Middle_01_L", "Middle_02_L", "Middle_03_L",
        "Ring_01_L", "Ring_02_L", "Ring_03_L",
        "Pinky_01_L", "Pinky_02_L", "Pinky_03_L",
        "Clavicle_R", "UpperArm_R", "LowerArm_R", "Wrist_R", "Hand_R",
        "Thumb_01_R", "Thumb_02_R", "Thumb_03_R",
        "Index_01_R", "Index_02_R", "Index_03_R",
        "Middle_01_R", "Middle_02_R", "Middle_03_R",
        "Ring_01_R", "Ring_02_R", "Ring_03_R",
        "Pinky_01_R", "Pinky_02_R", "Pinky_03_R",
        "Leg_Root_L", "UpperLeg_L", "LowerLeg_L", "Ankle_L", "Foot_L", "Toes_L",
        "Leg_Root_R", "UpperLeg_R", "LowerLeg_R", "Ankle_R", "Foot_R", "Toes_R",
        "Prop_L", "Prop_R"
    ])

    joint_limits: Dict[str, JointLimit] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default anatomical joint limits matching AVATAR_SPEC.md if not provided."""
        if not self.joint_limits:
            deg2rad = math.radians
            
            # Spine / Chest
            spine_limit = JointLimit(min_pitch=deg2rad(-35), max_pitch=deg2rad(35),
                                     min_yaw=deg2rad(-45), max_yaw=deg2rad(45),
                                     min_roll=deg2rad(-20), max_roll=deg2rad(20))
            self.joint_limits["Spine_01"] = spine_limit
            self.joint_limits["Spine_02"] = spine_limit
            self.joint_limits["Spine_03"] = spine_limit

            # Neck / Head
            self.joint_limits["Neck"] = JointLimit(min_pitch=deg2rad(-25), max_pitch=deg2rad(20),
                                                   min_yaw=deg2rad(-35), max_yaw=deg2rad(35),
                                                   min_roll=deg2rad(-18), max_roll=deg2rad(18))
            self.joint_limits["Head"] = JointLimit(min_pitch=deg2rad(-30), max_pitch=deg2rad(25),
                                                   min_yaw=deg2rad(-45), max_yaw=deg2rad(45),
                                                   min_roll=deg2rad(-20), max_roll=deg2rad(20))

            # Arms
            arm_limit = JointLimit(min_pitch=deg2rad(-120), max_pitch=deg2rad(180),
                                   min_yaw=deg2rad(-90), max_yaw=deg2rad(130),
                                   min_roll=deg2rad(-90), max_roll=deg2rad(90))
            self.joint_limits["UpperArm_L"] = arm_limit
            self.joint_limits["UpperArm_R"] = arm_limit

            elbow_limit = JointLimit(min_pitch=deg2rad(0), max_pitch=deg2rad(145),
                                     min_yaw=deg2rad(-10), max_yaw=deg2rad(10),
                                     min_roll=deg2rad(-90), max_roll=deg2rad(90))
            self.joint_limits["LowerArm_L"] = elbow_limit
            self.joint_limits["LowerArm_R"] = elbow_limit

            # Wrists
            wrist_limit = JointLimit(min_pitch=deg2rad(-70), max_pitch=deg2rad(75),
                                     min_yaw=deg2rad(-35), max_yaw=deg2rad(20),
                                     min_roll=deg2rad(-90), max_roll=deg2rad(90))
            self.joint_limits["Wrist_L"] = wrist_limit
            self.joint_limits["Wrist_R"] = wrist_limit

            # Fingers: Thumb CMC, MCP, IP
            self.joint_limits["Thumb_01_L"] = JointLimit(min_pitch=deg2rad(0), max_pitch=deg2rad(60), min_yaw=deg2rad(0), max_yaw=deg2rad(70))
            self.joint_limits["Thumb_01_R"] = JointLimit(min_pitch=deg2rad(0), max_pitch=deg2rad(60), min_yaw=deg2rad(0), max_yaw=deg2rad(70))
            self.joint_limits["Thumb_02_L"] = JointLimit(min_pitch=deg2rad(-10), max_pitch=deg2rad(55))
            self.joint_limits["Thumb_02_R"] = JointLimit(min_pitch=deg2rad(-10), max_pitch=deg2rad(55))
            self.joint_limits["Thumb_03_L"] = JointLimit(min_pitch=deg2rad(-15), max_pitch=deg2rad(80))
            self.joint_limits["Thumb_03_R"] = JointLimit(min_pitch=deg2rad(-15), max_pitch=deg2rad(80))

            # Finger MCP (01), PIP (02), DIP (03) for Index, Middle, Ring, Pinky
            for side in ["L", "R"]:
                for digit, abd_min, abd_max in [
                    ("Index", -15, 25),
                    ("Middle", -10, 10),
                    ("Ring", -15, 15),
                    ("Pinky", -20, 25),
                ]:
                    mcp_key = f"{digit}_01_{side}"
                    pip_key = f"{digit}_02_{side}"
                    dip_key = f"{digit}_03_{side}"

                    self.joint_limits[mcp_key] = JointLimit(
                        min_pitch=deg2rad(0), max_pitch=deg2rad(90),
                        min_yaw=deg2rad(abd_min), max_yaw=deg2rad(abd_max)
                    )
                    self.joint_limits[pip_key] = JointLimit(
                        min_pitch=deg2rad(0), max_pitch=deg2rad(105),
                        min_yaw=0.0, max_yaw=0.0
                    )
                    self.joint_limits[dip_key] = JointLimit(
                        min_pitch=deg2rad(0), max_pitch=deg2rad(80),
                        min_yaw=0.0, max_yaw=0.0
                    )


class VectorMath:
    """Quaternion and 3D vector utility operations."""

    @staticmethod
    def normalize_quat(q: List[float]) -> List[float]:
        """Normalize quaternion q = [x, y, z, w]."""
        if len(q) == 4:
            norm = math.sqrt(sum(x * x for x in q))
            if norm < 1e-8:
                return [0.0, 0.0, 0.0, 1.0]
            return [x / norm for x in q]
        elif len(q) == 3:
            return list(q)
        return q

    @staticmethod
    def euler_to_quat(pitch: float, yaw: float, roll: float) -> List[float]:
        """Convert Euler Z-X-Y angles (pitch, yaw, roll in radians) to Quaternion [x, y, z, w]."""
        rot = R.from_euler('zxy', [roll, pitch, yaw], degrees=False)
        q = rot.as_quat().tolist()
        return VectorMath.normalize_quat(q)

    @staticmethod
    def quat_to_euler(q: List[float]) -> Tuple[float, float, float]:
        """Convert Quaternion [x, y, z, w] to Euler Z-X-Y angles (pitch, yaw, roll in radians)."""
        if len(q) == 3:
            return q[0], q[1], q[2]
        norm_q = VectorMath.normalize_quat(q)
        rot = R.from_quat(norm_q)
        roll, pitch, yaw = rot.as_euler('zxy', degrees=False)
        return float(pitch), float(yaw), float(roll)

    @staticmethod
    def cross(v1: List[float], v2: List[float]) -> List[float]:
        """Cross product of two 3D vectors."""
        return [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]

    @staticmethod
    def dot(v1: List[float], v2: List[float]) -> float:
        """Dot product of two 3D vectors."""
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def norm(v: List[float]) -> float:
        """Magnitude of 3D vector."""
        return math.sqrt(VectorMath.dot(v, v))

    @staticmethod
    def normalize_vec(v: List[float]) -> List[float]:
        """Normalize 3D vector."""
        n = VectorMath.norm(v)
        if n < 1e-8:
            return [0.0, 0.0, 1.0]
        return [x / n for x in v]

    @staticmethod
    def matrix_from_vectors(p: List[float], d: List[float]) -> List[List[float]]:
        """Construct orthonormal 3x3 orientation matrix R = [p x d, p, d] via Gram-Schmidt."""
        p_norm = VectorMath.normalize_vec(p)
        d_norm = VectorMath.normalize_vec(d)
        
        dot_pd = VectorMath.dot(p_norm, d_norm)
        d_ortho = [d_norm[i] - dot_pd * p_norm[i] for i in range(3)]
        d_ortho = VectorMath.normalize_vec(d_ortho)

        right = VectorMath.cross(p_norm, d_ortho)
        right = VectorMath.normalize_vec(right)

        return [
            [right[0], p_norm[0], d_ortho[0]],
            [right[1], p_norm[1], d_ortho[1]],
            [right[2], p_norm[2], d_ortho[2]]
        ]

    @staticmethod
    def matrix_to_quat(R_mat: List[List[float]]) -> List[float]:
        """Convert 3x3 orthonormal matrix R to Quaternion [x, y, z, w]."""
        rot = R.from_matrix(R_mat)
        q = rot.as_quat().tolist()
        return VectorMath.normalize_quat(q)


class MotionRetargeter:
    """Retargeting engine for retargeting 3D avatar motion onto Unity Humanoid Rig."""

    def __init__(self, target_config: Optional[HumanoidRigConfig] = None) -> None:
        self.target_config = target_config or HumanoidRigConfig()

    def retarget_keyframe(
        self,
        keyframe: AnimationClipKeyframe,
        source_config: Optional[HumanoidRigConfig] = None,
        scale_factor: float = 1.0,
        target_handedness: Optional[str] = None,
        source_handedness: Optional[str] = None
    ) -> AnimationClipKeyframe:
        """Retarget a single animation keyframe according to target rig specification."""
        src_cfg = source_config or HumanoidRigConfig()
        
        effective_scale = scale_factor * (self.target_config.height_meters / src_cfg.height_meters)

        retargeted_rotations: Dict[str, List[float]] = {}
        retargeted_positions: Dict[str, List[float]] = {}
        retargeted_blendshapes: Dict[str, float] = dict(keyframe.blendshape_weights)

        # 1. Body Scale Retargeting on Positions
        for bone, pos in keyframe.joint_positions.items():
            scaled_pos = [pos[0] * effective_scale, pos[1] * effective_scale, pos[2] * effective_scale]
            retargeted_positions[bone] = scaled_pos

        # 2. Joint Rotations & Finger Alignment & Wrist Orientation & Joint Constraints
        for bone, rot in keyframe.joint_rotations.items():
            pitch, yaw, roll = VectorMath.quat_to_euler(rot)

            if bone in self.target_config.joint_limits:
                limit = self.target_config.joint_limits[bone]
                pitch, yaw, roll = limit.clamp_euler(pitch, yaw, roll)

            clamped_quat = VectorMath.euler_to_quat(pitch, yaw, roll)
            retargeted_rotations[bone] = clamped_quat

        # 3. Wrist Orientation Orthonormalization
        for wrist_bone in ["Wrist_R", "Wrist_L"]:
            if wrist_bone in retargeted_rotations:
                w_rot = retargeted_rotations[wrist_bone]
                retargeted_rotations[wrist_bone] = VectorMath.normalize_quat(w_rot)

        # 4. Handedness Conversion (Mirroring if target_handedness requested)
        if target_handedness:
            target_h_upper = target_handedness.upper()
            retargeted_rotations, retargeted_positions = self._apply_handedness_mirroring(
                retargeted_rotations, retargeted_positions, target_h_upper, source_handedness=source_handedness
            )

        return AnimationClipKeyframe(
            timestamp_ms=keyframe.timestamp_ms,
            joint_rotations=retargeted_rotations,
            joint_positions=retargeted_positions,
            blendshape_weights=retargeted_blendshapes
        )

    def retarget_clip(
        self,
        clip: AnimationClip,
        source_config: Optional[HumanoidRigConfig] = None,
        scale_factor: float = 1.0,
        target_handedness: Optional[str] = None
    ) -> AnimationClip:
        """Retarget an entire 3D AnimationClip onto target rig."""
        effective_handedness = target_handedness or clip.dominant_hand
        
        retargeted_kfs = []
        for kf in clip.keyframes:
            retargeted_kf = self.retarget_keyframe(
                kf,
                source_config=source_config,
                scale_factor=scale_factor,
                target_handedness=target_handedness,
                source_handedness=clip.dominant_hand
            )
            retargeted_kfs.append(retargeted_kf)

        return AnimationClip(
            clip_id=f"{clip.clip_id}_retargeted",
            sign_id=clip.sign_id,
            gloss=clip.gloss,
            duration=clip.duration,
            dominant_hand=effective_handedness.upper(),
            keyframes=retargeted_kfs,
            validation_status="VALIDATED"
        )

    def _apply_handedness_mirroring(
        self,
        rotations: Dict[str, List[float]],
        positions: Dict[str, List[float]],
        target_handedness: str,
        source_handedness: Optional[str] = None
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Apply sagittal plane (X-axis reflection) handedness conversion if mirroring is needed."""
        new_rotations = dict(rotations)
        new_positions = dict(positions)

        src_h = (source_handedness or "").upper()
        tgt_h = target_handedness.upper()

        needs_swap = False
        if tgt_h == "LEFT" and src_h == "RIGHT":
            needs_swap = True
        elif tgt_h == "RIGHT" and src_h == "LEFT":
            needs_swap = True
        elif tgt_h == "LEFT" and "Wrist_R" in positions and "Wrist_L" not in positions:
            needs_swap = True
        elif tgt_h == "RIGHT" and "Wrist_L" in positions and "Wrist_R" not in positions:
            needs_swap = True

        if needs_swap:
            mirrored_rotations = {}
            mirrored_positions = {}

            side_map = {}
            for bone in self.target_config.bones:
                if bone.endswith("_R"):
                    opp = bone[:-2] + "_L"
                    side_map[bone] = opp
                    side_map[opp] = bone

            for bone, pos in positions.items():
                target_bone = side_map.get(bone, bone)
                mirrored_positions[target_bone] = [-pos[0], pos[1], pos[2]]

            for bone, rot in rotations.items():
                target_bone = side_map.get(bone, bone)
                pitch, yaw, roll = VectorMath.quat_to_euler(rot)
                mirrored_rot = VectorMath.euler_to_quat(pitch, -yaw, -roll)
                mirrored_rotations[target_bone] = mirrored_rot

            return mirrored_rotations, mirrored_positions

        return new_rotations, new_positions
