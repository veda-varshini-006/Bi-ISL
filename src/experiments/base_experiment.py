"""Base Experiment Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExperiment(ABC):
    """Abstract base class for all research experiments (E0-E10)."""
    
    def __init__(self, exp_id: str, title: str, config_path: str):
        self.exp_id = exp_id
        self.title = title
        self.config_path = config_path
        
    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def teardown(self) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
