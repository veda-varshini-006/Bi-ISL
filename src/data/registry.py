"""Dataset Registry Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.data.schema import CanonicalDataSample

class BaseDatasetRegistry(ABC):
    """Abstract interface for registering and managing ISL dataset benchmarks."""
    
    @abstractmethod
    def register_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def get_sample(self, dataset_name: str, sample_id: str) -> CanonicalDataSample:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def list_datasets(self) -> List[str]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
