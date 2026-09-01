"""Controlled-Domain English-to-ISL Generator Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseStructuredGenerator(ABC):
    """Abstract interface for structured English-to-ISL IR generation (ADR-007)."""
    
    @abstractmethod
    def generate_isl_representation(self, english_input: str, dialogue_state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
