"""Uncertainty Estimator Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseUncertaintyEstimator(ABC):
    """Abstract interface for calculating predictive entropy/variance uncertainty."""
    
    @abstractmethod
    def estimate_uncertainty(self, model_output: Any) -> float:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
