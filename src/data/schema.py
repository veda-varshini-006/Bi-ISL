"""Canonical Data Sample Schema for Bi-ISL."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class CanonicalDataSample:
    sample_id: str
    signer_id: str
    video_path: Optional[str] = None
    frame_count: int = 0
    fps: float = 30.0
    text_target: str = ""
    gloss_target: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
