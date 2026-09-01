"""Loss Modules Interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLossModule(ABC):
    """Abstract interface for translation and alignment losses."""
    
    @abstractmethod
    def compute_loss(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
