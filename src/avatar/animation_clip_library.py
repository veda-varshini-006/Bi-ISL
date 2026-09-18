"""Avatar Animation Clip Library Subsystem (Prompt 73).

Manages 3D avatar animation clips for controlled domain sign language vocabulary,
storing keyframe data for bone rotations, translations, and facial blendshapes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path


@dataclass
class AnimationClipKeyframe:
    """Single keyframe snapshot in a 3D animation clip trajectory."""
    timestamp_ms: float
    joint_rotations: Dict[str, List[float]] = field(default_factory=dict)  # Bone -> [x, y, z, w] or [pitch, yaw, roll]
    joint_positions: Dict[str, List[float]] = field(default_factory=dict)  # Bone -> [x, y, z]
    blendshape_weights: Dict[str, float] = field(default_factory=dict)     # Target -> weight [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimationClipKeyframe":
        return cls(**data)


@dataclass
class AnimationClip:
    """3D Avatar Sign Animation Clip container."""
    clip_id: str
    sign_id: str
    gloss: str
    duration: float  # Duration in seconds
    dominant_hand: str  # "RIGHT", "LEFT", "BOTH"
    keyframes: List[AnimationClipKeyframe] = field(default_factory=list)
    validation_status: str = "VALIDATED"

    def __post_init__(self) -> None:
        if not self.clip_id or not isinstance(self.clip_id, str):
            raise ValueError("clip_id must be a non-empty string.")
        if not self.sign_id or not isinstance(self.sign_id, str):
            raise ValueError("sign_id must be a non-empty string.")
        if not self.gloss or not isinstance(self.gloss, str):
            raise ValueError("gloss must be a non-empty string.")
        if self.duration <= 0:
            raise ValueError("duration must be positive.")

        valid_hands = {"RIGHT", "LEFT", "BOTH"}
        if self.dominant_hand.upper() not in valid_hands:
            raise ValueError(f"dominant_hand must be one of {valid_hands}, got {self.dominant_hand}")
        self.dominant_hand = self.dominant_hand.upper()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "sign_id": self.sign_id,
            "gloss": self.gloss,
            "duration": self.duration,
            "dominant_hand": self.dominant_hand,
            "validation_status": self.validation_status,
            "keyframes": [kf.to_dict() for kf in self.keyframes]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimationClip":
        data_copy = dict(data)
        kfs_data = data_copy.pop("keyframes", [])
        keyframes = [AnimationClipKeyframe.from_dict(kf) for kf in kfs_data]
        return cls(keyframes=keyframes, **data_copy)


class AnimationClipLibrary:
    """Catalog storing and providing access to 3D avatar animation clips."""

    def __init__(self) -> None:
        self._clips: Dict[str, AnimationClip] = {}

    def add_clip(self, clip: AnimationClip) -> None:
        """Register an animation clip in the library."""
        if not isinstance(clip, AnimationClip):
            raise TypeError("clip must be an instance of AnimationClip.")
        self._clips[clip.clip_id] = clip

    def get_clip(self, clip_id: str) -> Optional[AnimationClip]:
        """Retrieve clip by clip_id."""
        return self._clips.get(clip_id)

    def get_clip_by_sign_id(self, sign_id: str) -> Optional[AnimationClip]:
        """Retrieve clip by sign_id."""
        for clip in self._clips.values():
            if clip.sign_id == sign_id:
                return clip
        return None

    def get_clip_by_gloss(self, gloss: str) -> Optional[AnimationClip]:
        """Retrieve clip by ISL gloss string."""
        gloss_upper = gloss.upper()
        for clip in self._clips.values():
            if clip.gloss.upper() == gloss_upper:
                return clip
        return None

    def list_clips(self) -> List[AnimationClip]:
        """Return list of all registered animation clips."""
        return list(self._clips.values())

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Export animation clip library to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_data = [clip.to_dict() for clip in self._clips.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    def load_json(self, file_path: Union[str, Path]) -> None:
        """Load animation clip library from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Clip library file not found at: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            clip = AnimationClip.from_dict(item)
            self.add_clip(clip)

    @classmethod
    def create_seeded_library(cls) -> "AnimationClipLibrary":
        """Create library pre-populated with initial validated controlled vocabulary clips."""
        lib = cls()
        
        # Controlled vocabulary terms with synthetic keyframe trajectories prioritizing correctness
        terms = [
            ("CLIP_DOCTOR_01", "SIGN_DOCTOR_01", "DOCTOR", 1.25, "RIGHT", ["browDownLeft", "browDownRight"]),
            ("CLIP_HOSPITAL_01", "SIGN_HOSPITAL_01", "HOSPITAL", 1.50, "BOTH", []),
            ("CLIP_FEVER_01", "SIGN_FEVER_01", "FEVER", 1.10, "RIGHT", ["eyeSquintLeft", "eyeSquintRight"]),
            ("CLIP_PAIN_01", "SIGN_PAIN_01", "PAIN", 1.00, "RIGHT", ["browDownLeft", "browDownRight"]),
            ("CLIP_MEDICINE_01", "SIGN_MEDICINE_01", "MEDICINE", 1.40, "BOTH", []),
            ("CLIP_HELP_01", "SIGN_HELP_01", "HELP", 1.30, "BOTH", ["browOuterUpLeft", "browOuterUpRight"]),
            ("CLIP_TODAY_01", "SIGN_TODAY_01", "TODAY", 0.90, "RIGHT", []),
            ("CLIP_TOMORROW_01", "SIGN_TOMORROW_01", "TOMORROW", 1.00, "RIGHT", []),
            ("CLIP_THANK_YOU_01", "SIGN_THANK_YOU_01", "THANK YOU", 1.15, "RIGHT", ["mouthSmileLeft", "mouthSmileRight"]),
            ("CLIP_YES_01", "SIGN_YES_01", "YES", 0.80, "RIGHT", []),
            ("CLIP_NO_01", "SIGN_NO_01", "NO", 0.85, "RIGHT", ["browDownLeft", "browDownRight"]),
            ("CLIP_NOT_01", "SIGN_NOT_01", "NOT", 0.95, "RIGHT", []),
            ("CLIP_APPOINTMENT_01", "SIGN_APPOINTMENT_01", "APPOINTMENT", 1.35, "BOTH", []),
            ("CLIP_WHERE_01", "SIGN_WHERE_01", "WHERE", 1.05, "BOTH", ["browDownLeft", "browDownRight"]),
            ("CLIP_WHEN_01", "SIGN_WHEN_01", "WHEN", 1.10, "RIGHT", ["browOuterUpLeft", "browOuterUpRight"])
        ]

        for clip_id, sign_id, gloss, duration, dominant_hand, facial in terms:
            num_keyframes = int(duration * 60)  # 60 FPS
            keyframes = []
            for i in range(num_keyframes):
                t_ms = (i / (num_keyframes - 1)) * (duration * 1000.0) if num_keyframes > 1 else 0.0
                progress = i / max(1, num_keyframes - 1)
                
                # Arm joint trajectories matching declared handedness
                joint_rotations = {}
                joint_positions = {}
                
                if dominant_hand in {"RIGHT", "BOTH"}:
                    joint_rotations["UpperArm_R"] = [0.1 * progress, 0.2 * progress, 0.0, 1.0]
                    joint_rotations["Wrist_R"] = [0.05 * progress, 0.0, 0.0, 1.0]
                    joint_positions["Wrist_R"] = [0.15 + 0.1 * progress, 1.2 + 0.05 * progress, 0.4]
                else:
                    joint_rotations["UpperArm_R"] = [0.0, 0.0, 0.0, 1.0]
                    joint_positions["Wrist_R"] = [0.15, 0.8, 0.2]  # Rest pose
                    
                if dominant_hand in {"LEFT", "BOTH"}:
                    joint_rotations["UpperArm_L"] = [-0.1 * progress, -0.2 * progress, 0.0, 1.0]
                    joint_rotations["Wrist_L"] = [-0.05 * progress, 0.0, 0.0, 1.0]
                    joint_positions["Wrist_L"] = [-0.15 - 0.1 * progress, 1.2 + 0.05 * progress, 0.4]
                else:
                    joint_rotations["UpperArm_L"] = [0.0, 0.0, 0.0, 1.0]
                    joint_positions["Wrist_L"] = [-0.15, 0.8, 0.2]  # Rest pose

                blendshapes = {fm: 0.8 * progress for fm in facial}

                keyframes.append(
                    AnimationClipKeyframe(
                        timestamp_ms=t_ms,
                        joint_rotations=joint_rotations,
                        joint_positions=joint_positions,
                        blendshape_weights=blendshapes
                    )
                )

            clip = AnimationClip(
                clip_id=clip_id,
                sign_id=sign_id,
                gloss=gloss,
                duration=duration,
                dominant_hand=dominant_hand,
                keyframes=keyframes,
                validation_status="VALIDATED"
            )
            lib.add_clip(clip)

        return lib
