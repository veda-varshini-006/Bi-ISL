"""Avatar Renderer Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseAvatarRenderer(ABC):
    """Abstract interface for 3D avatar rendering engine (ADR-008)."""
    
    @abstractmethod
    def render_frame_sequence(self, motion_keyframes: Any, blendshapes: Any) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
