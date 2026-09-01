"""Translation Quality Metrics Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTranslationMetric(ABC):
    """Abstract interface for calculating BLEU-4, chrF++, and BERTScore."""
    
    @abstractmethod
    def compute_score(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
