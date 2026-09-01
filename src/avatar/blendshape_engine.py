"""Facial Blendshape Engine Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseBlendshapeEngine(ABC):
    """Abstract interface for driving facial blendshapes for non-manual markers."""
    
    @abstractmethod
    def compute_facial_blendshapes(self, non_manual_tags: List[Dict[str, Any]]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
