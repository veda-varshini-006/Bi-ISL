"""NMM Debug Overlay Subsystem (Prompt 76).

Formats on-screen HUD frame metadata displaying active NMM tags, eyebrow state,
blendshape weights, head pose, body lean, and frame timestamps.
"""

import math
from typing import Dict, Any, List, Optional
from src.avatar.nmm_controller import NMMFrameState


class NMMDebugOverlay:
    """Debug Overlay renderer for Non-Manual Marker HUD visualization."""

    @staticmethod
    def format_hud_text(nmm_state: NMMFrameState, frame_index: int = 0) -> str:
        """Format clean multiline HUD text for display on screen or in logs."""
        rad2deg = math.degrees

        hp_deg = [rad2deg(a) for a in nmm_state.head_pose_euler]
        bp_deg = [rad2deg(a) for a in nmm_state.body_lean_euler]

        tags_str = ", ".join(nmm_state.active_tags) if nmm_state.active_tags else "NONE"
        
        active_bs = [
            f"{k}: {v:.2f}" for k, v in nmm_state.blendshapes.items() if v > 0.01
        ]
        bs_str = ", ".join(active_bs) if active_bs else "NONE"

        lines = [
            "=== ISL AVATAR NMM DEBUG OVERLAY ===",
            f"Timestamp   : {nmm_state.timestamp_ms:.1f} ms (Frame: {frame_index})",
            f"Active NMMs : [{tags_str}]",
            f"Eyebrow     : {nmm_state.eyebrow_state}",
            f"Blendshapes : [{bs_str}]",
            f"Head Pose   : Pitch: {hp_deg[0]:+.1f}°, Yaw: {hp_deg[1]:+.1f}°, Roll: {hp_deg[2]:+.1f}°",
            f"Body Lean   : Pitch: {bp_deg[0]:+.1f}°, Yaw: {bp_deg[1]:+.1f}°, Roll: {bp_deg[2]:+.1f}°",
            "===================================="
        ]

        return "\n".join(lines)

    @staticmethod
    def generate_overlay_frame_data(nmm_state: NMMFrameState, frame_index: int = 0) -> Dict[str, Any]:
        """Generate structured JSON payload for Unity or web canvas debug HUD rendering."""
        rad2deg = math.degrees
        return {
            "frame_index": frame_index,
            "timestamp_ms": nmm_state.timestamp_ms,
            "active_tags": nmm_state.active_tags,
            "active_tag_count": len(nmm_state.active_tags),
            "eyebrow_state": nmm_state.eyebrow_state,
            "active_blendshapes": {k: round(v, 3) for k, v in nmm_state.blendshapes.items() if v > 0.01},
            "head_pose_deg": {
                "pitch": round(rad2deg(nmm_state.head_pose_euler[0]), 2),
                "yaw": round(rad2deg(nmm_state.head_pose_euler[1]), 2),
                "roll": round(rad2deg(nmm_state.head_pose_euler[2]), 2),
            },
            "body_lean_deg": {
                "pitch": round(rad2deg(nmm_state.body_lean_euler[0]), 2),
                "yaw": round(rad2deg(nmm_state.body_lean_euler[1]), 2),
                "roll": round(rad2deg(nmm_state.body_lean_euler[2]), 2),
            },
            "hud_text": NMMDebugOverlay.format_hud_text(nmm_state, frame_index=frame_index)
        }
