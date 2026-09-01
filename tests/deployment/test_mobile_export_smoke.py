"""Smoke tests for deployment and ONNX export."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

class SimpleMobileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x)

def test_onnx_export_smoke():
    model = SimpleMobileModel()
    model.eval()

    dummy_input = torch.randn(1, 64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "model.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=14
        )

        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
