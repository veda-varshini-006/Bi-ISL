"""Deployment & System Benchmarking Metrics Interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDeploymentMetric(ABC):
    """Abstract interface for measuring p50/p95 latency, peak RAM, and storage size."""
    
    @abstractmethod
    def compute_deployment_metrics(self, latency_records: List[float], ram_records: List[float]) -> Dict[str, float]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
