"""Split Auditor Interface for Experiment E0."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.data.schema import CanonicalDataSample

class BaseSplitAuditor(ABC):
    """Abstract interface for dataset leakage audit (Experiment E0)."""
    
    @abstractmethod
    def audit_splits(self, train_samples: List[CanonicalDataSample], dev_samples: List[CanonicalDataSample], test_samples: List[CanonicalDataSample]) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
