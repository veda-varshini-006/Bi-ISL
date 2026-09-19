"""Unity ISL Animation Payload Exporter Subsystem (Prompt 77).

Exports complete pre-computed ISL intermediate representation animation keyframes,
retargeted bone quaternions, blendshape weights, and NMM debug state into
deterministic JSON payloads for Unity rendering with zero translation logic in Unity.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from src.avatar.motion_mapper import RetargetingMotionMapper
from src.avatar.nmm_controller import NMMController, NonManualTagSpec
from src.avatar.nmm_debug_overlay import NMMDebugOverlay


@dataclass
class UnityISLAnimationPayload:
    """Deterministic 3D avatar animation payload container for Unity rendering."""
    sequence_id: str
    gloss_sequence: List[str]
    total_duration_sec: float
    fps: float = 60.0
    total_frames: int = 0
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnityISLAnimationPayload":
        return cls(**data)

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "UnityISLAnimationPayload":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class UnityPayloadExporter:
    """Exporter generating deterministic JSON animation payloads for Unity rendering."""

    def __init__(self, mapper: Optional[RetargetingMotionMapper] = None) -> None:
        self.mapper = mapper or RetargetingMotionMapper()
        self.nmm_controller = NMMController()

    def export_payload_from_ir_sequence(
        self,
        isl_ir_sequence: List[Dict[str, Any]],
        sequence_id: str = "SEQ_UNITY_PAYLOAD_01",
        output_json_path: Optional[Union[str, Path]] = None,
        default_transition_ms: float = 120.0,
        hold_duration_ms: float = 0.0
    ) -> UnityISLAnimationPayload:
        """Export pre-computed keyframes and NMM state matrix into a deterministic Unity payload."""
        if not isl_ir_sequence:
            raise ValueError("isl_ir_sequence cannot be empty.")

        glosses = [item.get("gloss") or item.get("sign_gloss") or "HELP" for item in isl_ir_sequence]

        # Map sequence with controlled transitions and NMM markers
        keyframes_data = self.mapper.map_sequence_with_transitions(
            isl_ir_sequence,
            default_transition_ms=default_transition_ms,
            hold_duration_ms=hold_duration_ms
        )

        total_frames = len(keyframes_data)
        total_duration_sec = (keyframes_data[-1]["timestamp_ms"] / 1000.0) if keyframes_data else 0.0

        # Enhance keyframes with NMM debug overlay metadata frame states
        enhanced_keyframes = []
        for idx, kf in enumerate(keyframes_data):
            ts = kf["timestamp_ms"]
            
            # Extract active NMM specs for frame overlay snapshot
            tag_specs = []
            for item in isl_ir_sequence:
                if "non_manual_markers" in item:
                    for nm_item in item["non_manual_markers"]:
                        tag = nm_item.get("tag") or nm_item.get("marker_type") or "neutral_facial"
                        tag_specs.append(
                            NonManualTagSpec(
                                tag=tag,
                                category=nm_item.get("category", "general"),
                                start_ms=float(nm_item.get("start_ms", 0.0)),
                                end_ms=float(nm_item.get("end_ms", total_duration_sec * 1000.0)),
                                intensity=float(nm_item.get("intensity", 1.0))
                            )
                        )

            nmm_state = self.nmm_controller.evaluate_nmm_state(tag_specs, ts)
            overlay_data = NMMDebugOverlay.generate_overlay_frame_data(nmm_state, frame_index=idx)

            kf_enhanced = dict(kf)
            kf_enhanced["nmm_state"] = nmm_state.to_dict()
            kf_enhanced["debug_overlay"] = overlay_data
            enhanced_keyframes.append(kf_enhanced)

        payload = UnityISLAnimationPayload(
            sequence_id=sequence_id,
            gloss_sequence=glosses,
            total_duration_sec=total_duration_sec,
            fps=60.0,
            total_frames=total_frames,
            keyframes=enhanced_keyframes,
            metadata={
                "target_rig": "UnityHumanoidRig_Standard",
                "hand_count": 2,
                "finger_dof_count": 40,
                "has_nmm": True,
                "zero_unity_translation": True
            }
        )

        if output_json_path:
            payload.save_json(output_json_path)

        return payload
