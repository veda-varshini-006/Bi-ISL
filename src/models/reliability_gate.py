"""Context Reliability Gate Interface."""
from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseReliabilityGate(ABC):
    """Abstract interface for context-evidence reliability gating (ADR-005)."""
    
    @abstractmethod
    def compute_gate(self, visual_features: Any, context_features: Any) -> Tuple[Any, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
