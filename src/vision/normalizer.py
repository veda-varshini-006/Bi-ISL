"""Feature Normalizer Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseFeatureNormalizer(ABC):
    """Abstract interface for spatial and temporal feature normalization."""
    
    @abstractmethod
    def normalize(self, features: Any) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
