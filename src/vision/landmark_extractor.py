"""Landmark Extractor Interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLandmarkExtractor(ABC):
    """Abstract interface for extracting pose, hand, and facial keypoints."""
    
    @abstractmethod
    def extract_landmarks(self, video_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
