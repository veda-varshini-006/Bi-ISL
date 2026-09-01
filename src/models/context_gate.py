"""Context Encoder Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseContextEncoder(ABC):
    """Abstract interface for encoding dialogue history context."""
    
    @abstractmethod
    def encode_context(self, dialogue_state: Any) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
