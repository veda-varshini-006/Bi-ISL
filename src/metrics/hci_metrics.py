"""HCI & Avatar Evaluation Metrics Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseHCIMetric(ABC):
    """Abstract interface for analyzing DHH comprehension accuracy and SUS scores."""
    
    @abstractmethod
    def analyze_user_responses(self, survey_data: List[Dict[str, Any]]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
