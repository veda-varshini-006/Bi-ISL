"""Base Experiment Interface for Bi-ISL."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.utils.config import BiISLConfig
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logging import BiISLLogger


class BaseExperiment(ABC):
    """Abstract base class for all research experiments (E0-E10)."""

    def __init__(
        self,
        exp_id: str = "E0",
        title: str = "Experiment",
        config_path: Optional[str] = None,
        config: Optional[BiISLConfig] = None
    ):
        self.exp_id = exp_id
        self.experiment_id = exp_id
        self.title = title
        self.config_path = config_path
        self.config = config or BiISLConfig()
        self.logger = BiISLLogger(name=f"Experiment_{exp_id}")
        self.tracker = ExperimentTracker(experiment_id=exp_id, config=self.config)

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass
