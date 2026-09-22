"""Mobile Exporter Implementation for PyTorch to ONNX/ExecuTorch (ADR-009)."""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import torch.nn as nn


class BaseMobileExporter(ABC):
    """Abstract interface for exporting PyTorch models to ONNX / ExecuTorch (ADR-009)."""
    
    @abstractmethod
    def export_to_onnx(self, model: Any, output_path: str, dummy_input: Optional[torch.Tensor] = None) -> str:
        """Export PyTorch model to ONNX format."""
        pass
        
    @abstractmethod
    def quantize_model(self, onnx_path: str, output_path: str, precision: str = "int8") -> str:
        """Quantize an exported model file."""
        pass


class PyTorchMobileExporter(BaseMobileExporter):
    """Concrete implementation for PyTorch ONNX export and mobile quantization."""

    def export_to_onnx(self, model: Any, output_path: str, dummy_input: Optional[torch.Tensor] = None) -> str:
        """Export a PyTorch nn.Module to ONNX format."""
        if not isinstance(model, nn.Module):
            raise TypeError("Expected PyTorch nn.Module instance.")
            
        model.eval()
        if dummy_input is None:
            dummy_input = torch.randn(1, 64)
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        return output_path

    def quantize_model(self, onnx_path: str, output_path: str, precision: str = "int8") -> str:
        """Quantize exported ONNX model or PyTorch checkpoint file."""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model path not found: {onnx_path}")
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            if precision.lower() in ("int8", "int8_dynamic"):
                quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QUInt8)
            else:
                # Copy as fallback if specialized ONNX quantization isn't requested
                with open(onnx_path, "rb") as f_in, open(output_path, "wb") as f_out:
                    f_out.write(f_in.read())
        except Exception:
            # Fallback file copy if ONNX runtime is unavailable
            with open(onnx_path, "rb") as f_in, open(output_path, "wb") as f_out:
                f_out.write(f_in.read())

        return output_path
