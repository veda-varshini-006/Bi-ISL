"""Entity Tracker Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEntityTracker(ABC):
    """Abstract interface for tracking dialogue entities and referents across turns."""
    
    @abstractmethod
    def extract_entities(self, text_or_gloss: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
