"""RGB Feature Extractor Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseRGBExtractor(ABC):
    """Abstract interface for spatiotemporal RGB feature extraction."""
    
    @abstractmethod
    def extract_features(self, video_path: str) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
