"""Facial Blendshape Engine Subsystem (Prompt 76).

Computes 32 FACS/ISL facial blendshapes from non-manual marker specifications.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.avatar.nmm_controller import NMMController, NonManualTagSpec


class BaseBlendshapeEngine(ABC):
    """Abstract interface for driving facial blendshapes for non-manual markers."""
    
    @abstractmethod
    def compute_facial_blendshapes(self, non_manual_tags: List[Dict[str, Any]]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")


class NMMBlendshapeEngine(BaseBlendshapeEngine):
    """Engine computing facial blendshapes for non-manual markers."""

    def __init__(self) -> None:
        self.controller = NMMController()

    def compute_facial_blendshapes(self, non_manual_tags: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute 32 FACS facial blendshape weights from non-manual marker dictionaries."""
        specs = []
        for item in non_manual_tags:
            tag = item.get("tag") or item.get("marker_type") or "neutral_facial"
            category = item.get("category", "general")
            start_ms = float(item.get("start_ms", 0.0))
            end_ms = float(item.get("end_ms", 1000.0))
            intensity = float(item.get("intensity", 1.0))
            attack_ms = float(item.get("attack_ms", 100.0))
            decay_ms = float(item.get("decay_ms", 80.0))

            specs.append(
                NonManualTagSpec(
                    tag=tag,
                    category=category,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    intensity=intensity,
                    attack_ms=attack_ms,
                    decay_ms=decay_ms
                )
            )

        mid_ts = 500.0  # Sample mid-point timestamp for static frame calculation
        state = self.controller.evaluate_nmm_state(specs, mid_ts)
        return state.blendshapes
