"""Motion Sequence Mapper Subsystem (Prompt 74).

Maps ISL IR tokens into retargeted skeletal motion keyframes onto Unity Humanoid Rigs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.avatar.animation_clip_library import AnimationClip, AnimationClipKeyframe, AnimationClipLibrary
from src.avatar.motion_retargeter import MotionRetargeter, HumanoidRigConfig


class BaseMotionMapper(ABC):
    """Abstract interface for mapping ISL IR tokens to skeletal keyframes."""
    
    @abstractmethod
    def map_to_motion_keyframes(self, isl_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")


class RetargetingMotionMapper(BaseMotionMapper):
    """Motion Mapper implementing retargeted keyframe generation for Unity Humanoid Rig."""

    def __init__(
        self,
        clip_library: Optional[AnimationClipLibrary] = None,
        target_config: Optional[HumanoidRigConfig] = None
    ) -> None:
        self.clip_library = clip_library or AnimationClipLibrary.create_seeded_library()
        self.retargeter = MotionRetargeter(target_config or HumanoidRigConfig())

    def map_to_motion_keyframes(self, isl_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map ISL IR representation to retargeted motion keyframe dictionaries."""
        gloss = isl_ir.get("gloss") or isl_ir.get("sign_gloss") or "HELP"
        clip = self.clip_library.get_clip_by_gloss(gloss)
        if clip is None:
            clip = self.clip_library.get_clip_by_gloss("HELP")

        if clip is None:
            return []

        scale = float(isl_ir.get("scale_factor", 1.0))
        target_handedness = isl_ir.get("handedness")

        retargeted_clip = self.retargeter.retarget_clip(
            clip, scale_factor=scale, target_handedness=target_handedness
        )

        return [kf.to_dict() for kf in retargeted_clip.keyframes]
