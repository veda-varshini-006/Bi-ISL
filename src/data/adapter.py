"""Dataset Adapter Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.data.schema import CanonicalDataSample

class BaseDatasetAdapter(ABC):
    """Abstract interface for converting raw benchmark data to CanonicalDataSample format."""
    
    @abstractmethod
    def load_raw_dataset(self, dataset_path: str) -> Any:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def parse_samples(self, raw_data: Any) -> List[CanonicalDataSample]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
