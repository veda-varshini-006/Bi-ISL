"""Non-Manual Marker Tagger Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseNonManualTagger(ABC):
    """Abstract interface for tagging non-manual facial/body grammatical markers."""
    
    @abstractmethod
    def tag_non_manuals(self, gloss_sequence: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
