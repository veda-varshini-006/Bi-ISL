"""Personalization Metrics Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePersonalizationMetric(ABC):
    """Abstract interface for calculating per-signer delta BLEU and ECE calibration."""
    
    @abstractmethod
    def compute_signer_metrics(self, per_signer_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
