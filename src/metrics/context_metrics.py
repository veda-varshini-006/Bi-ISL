"""Context Reliability Metrics Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseContextMetric(ABC):
    """Abstract interface for calculating Unsupported Slot Rate (USR)."""
    
    @abstractmethod
    def compute_unsupported_slot_rate(self, hypotheses: List[str], visual_ground_truth: List[Dict[str, Any]]) -> float:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
