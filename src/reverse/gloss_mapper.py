"""Gloss Mapper Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGlossMapper(ABC):
    """Abstract interface for mapping English text to ISL gloss ordering."""
    
    @abstractmethod
    def map_to_gloss_sequence(self, english_text: str) -> List[str]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
