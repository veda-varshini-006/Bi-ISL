"""Sequence Decoder Interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSequenceDecoder(ABC):
    """Abstract interface for sequence translation decoding."""
    
    @abstractmethod
    def decode(self, encoder_output: Any, context_gate_weight: float = 1.0) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
