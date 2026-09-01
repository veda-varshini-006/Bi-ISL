"""Shared Bidirectional Dialogue State (SBDS) Manager Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSBDSManager(ABC):
    """Abstract interface for Shared Bidirectional Dialogue State management (ADR-004)."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def update_state(self, turn_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def reset_state(self) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
