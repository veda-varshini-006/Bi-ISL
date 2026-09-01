"""Uncertainty-Gated Signer Adaptation (UGSA) Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseUGSAAdapter(ABC):
    """Abstract interface for uncertainty-gated online signer adaptation (ADR-006)."""
    
    @abstractmethod
    def adapt(self, model: Any, sample: Dict[str, Any], uncertainty_score: float) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
