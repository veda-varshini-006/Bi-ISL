"""Avatar Renderer Subsystem (Prompt 77).

Implements deterministic frame rendering and playback controls:
- play()
- pause()
- repeat() / set_repeat()
- speed / set_speed()
- step_forward() / step_backward() / seek_to_frame()
- debug_mode / set_debug_mode()
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.avatar.unity_payload_exporter import UnityISLAnimationPayload


class BaseAvatarRenderer(ABC):
    """Abstract interface for 3D avatar rendering engine (ADR-008)."""
    
    @abstractmethod
    def render_frame_sequence(self, motion_keyframes: Any, blendshapes: Any) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")


class DeterministicAvatarRenderer(BaseAvatarRenderer):
    """3D Avatar Renderer implementing deterministic playback API controls."""

    def __init__(self, payload: Optional[UnityISLAnimationPayload] = None) -> None:
        self.payload: Optional[UnityISLAnimationPayload] = payload
        self.current_frame_index: int = 0
        self.is_playing: bool = False
        self.is_repeat_enabled: bool = False
        self.playback_speed: float = 1.0  # multiplier (e.g. 0.5x, 1.0x, 2.0x)
        self.is_debug_mode_enabled: bool = False

    def load_payload(self, payload: UnityISLAnimationPayload) -> None:
        """Load deterministic animation payload."""
        if not isinstance(payload, UnityISLAnimationPayload):
            raise TypeError("payload must be an instance of UnityISLAnimationPayload.")
        self.payload = payload
        self.current_frame_index = 0
        self.is_playing = False

    def play(self) -> None:
        """Start or resume playback."""
        if self.payload and self.payload.total_frames > 0:
            self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def set_repeat(self, repeat: bool) -> None:
        """Enable or disable loop repeat playback."""
        self.is_repeat_enabled = bool(repeat)

    def set_speed(self, speed_multiplier: float) -> None:
        """Set playback speed multiplier (e.g., 0.5x, 1.0x, 2.0x)."""
        if speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be positive.")
        self.playback_speed = float(speed_multiplier)

    def seek_to_frame(self, frame_index: int) -> Dict[str, Any]:
        """Seek directly to frame index and return frame data."""
        if not self.payload or self.payload.total_frames == 0:
            return {}
        max_idx = self.payload.total_frames - 1
        self.current_frame_index = max(0, min(max_idx, frame_index))
        return self.get_current_frame()

    def step_forward(self) -> Dict[str, Any]:
        """Advance playback by 1 frame."""
        if not self.payload or self.payload.total_frames == 0:
            return {}
        
        if self.current_frame_index < self.payload.total_frames - 1:
            self.current_frame_index += 1
        elif self.is_repeat_enabled:
            self.current_frame_index = 0

        return self.get_current_frame()

    def step_backward(self) -> Dict[str, Any]:
        """Step back playback by 1 frame."""
        if not self.payload or self.payload.total_frames == 0:
            return {}
        
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
        elif self.is_repeat_enabled:
            self.current_frame_index = self.payload.total_frames - 1

        return self.get_current_frame()

    def set_debug_mode(self, enabled: bool) -> None:
        """Toggle debug HUD overlay and gizmo visualization."""
        self.is_debug_mode_enabled = bool(enabled)

    def get_current_frame(self) -> Dict[str, Any]:
        """Get pre-computed frame data at current frame index."""
        if not self.payload or not self.payload.keyframes:
            return {}
        return self.payload.keyframes[self.current_frame_index]

    def render_frame_sequence(self, motion_keyframes: Any, blendshapes: Any) -> List[Dict[str, Any]]:
        """Render frame sequence deterministically."""
        if isinstance(motion_keyframes, UnityISLAnimationPayload):
            self.load_payload(motion_keyframes)
            return self.payload.keyframes if self.payload else []
        elif isinstance(motion_keyframes, list):
            return motion_keyframes
        return []
