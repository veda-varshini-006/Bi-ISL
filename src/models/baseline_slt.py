"""Baseline SLT Model Interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSLTModel(ABC):
    """Abstract interface for spatiotemporal SignLanguageTranslation research baseline."""
    
    @abstractmethod
    def forward(self, visual_inputs: Any, context_inputs: Any = None) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
