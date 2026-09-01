"""Experiment E10 Interface."""
from src.experiments.base_experiment import BaseExperiment
from typing import Dict, Any

class E10DHHUserStudyExperiment(BaseExperiment):
    """Abstract interface for Experiment E10: DHH User Comprehension Study."""
    
    def __init__(self, config_path: str):
        super().__init__("E10", "DHH User Comprehension Study", config_path)
        
    def setup() -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    def teardown(self) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
