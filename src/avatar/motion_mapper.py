"""Motion Sequence Mapper Subsystem (Prompt 74 & Prompt 75).

Maps ISL IR tokens into retargeted skeletal motion keyframes onto Unity Humanoid Rigs,
and synthesizes controlled transitions between consecutive sign clips.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.avatar.animation_clip_library import AnimationClip, AnimationClipKeyframe, AnimationClipLibrary
from src.avatar.motion_retargeter import MotionRetargeter, HumanoidRigConfig
from src.avatar.transition_engine import TransitionEngine, CoarticulationMetadata


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
        self.transition_engine = TransitionEngine()

    def map_to_motion_keyframes(self, isl_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map single ISL IR representation to retargeted motion keyframe dictionaries."""
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

    def map_sequence_with_transitions(
        self,
        isl_ir_sequence: List[Dict[str, Any]],
        default_transition_ms: float = 120.0,
        hold_duration_ms: float = 0.0,
        hold_handshape: bool = True
    ) -> List[Dict[str, Any]]:
        """Map sequence of ISL IR tokens into a stitched retargeted clip with controlled transitions."""
        if not isl_ir_sequence:
            return []

        clips_to_stitch: List[AnimationClip] = []
        meta_map: Dict[str, CoarticulationMetadata] = {}

        for item in isl_ir_sequence:
            gloss = item.get("gloss") or item.get("sign_gloss") or "HELP"
            clip = self.clip_library.get_clip_by_gloss(gloss)
            if clip is None:
                clip = self.clip_library.get_clip_by_gloss("HELP")

            if clip:
                scale = float(item.get("scale_factor", 1.0))
                target_handedness = item.get("handedness")
                retargeted = self.retargeter.retarget_clip(clip, scale_factor=scale, target_handedness=target_handedness)
                clips_to_stitch.append(retargeted)

                if "coarticulation" in item:
                    c_data = item["coarticulation"]
                    meta_map[retargeted.clip_id] = CoarticulationMetadata(
                        clip_id=retargeted.clip_id,
                        gloss=retargeted.gloss,
                        transition_duration_ms=c_data.get("transition_duration_ms", default_transition_ms),
                        hold_duration_ms=c_data.get("hold_duration_ms", hold_duration_ms),
                        hold_handshape=c_data.get("hold_handshape", hold_handshape),
                        blend_curve=c_data.get("blend_curve", "AHDR_HERMITE")
                    )

        if not clips_to_stitch:
            return []

        stitched_clip = self.transition_engine.stitch_clip_sequence(
            clips_to_stitch,
            coarticulation_metadata_map=meta_map,
            default_transition_ms=default_transition_ms,
            hold_duration_ms=hold_duration_ms,
            hold_handshape=hold_handshape
        )

        return [kf.to_dict() for kf in stitched_clip.keyframes]
