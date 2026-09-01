"""Context Reliability Estimator Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseContextReliabilityEstimator(ABC):
    """Abstract interface for estimating historical context reliability."""
    
    @abstractmethod
    def estimate_reliability(self, dialogue_state: Dict[str, Any], current_visual_confidence: float) -> float:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
