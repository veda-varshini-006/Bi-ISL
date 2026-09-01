"""Mobile Exporter Interface."""
from abc import ABC, abstractmethod
from typing import Any

class BaseMobileExporter(ABC):
    """Abstract interface for exporting PyTorch models to ONNX / ExecuTorch (ADR-009)."""
    
    @abstractmethod
    def export_to_onnx(self, model: Any, output_path: str) -> str:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
        
    @abstractmethod
    def quantize_model(self, onnx_path: str, output_path: str) -> str:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
