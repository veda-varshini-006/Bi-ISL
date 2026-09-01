"""Experiment E7 Interface."""
from src.experiments.base_experiment import BaseExperiment
from typing import Dict, Any

class E7GeneralizationExperiment(BaseExperiment):
    """Abstract interface for Experiment E7: Cross-Signer & Shift Generalization Experiment."""
    
    def __init__(self, config_path: str):
        super().__init__("E7", "Cross-Signer & Shift Generalization Experiment", config_path)
        
    def setup() -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    def teardown(self) -> None:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
