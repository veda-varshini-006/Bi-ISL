"""Unit tests for QuantizationEvaluator (Prompt 88)."""

import pytest
import torch
import torch.nn as nn
from src.deployment.quantization_evaluator import (
    QuantizationEvaluator,
    QuantizationVariantResult,
    QuantizationReport
)
from src.utils.config import BiISLConfig


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def evaluator():
    config = BiISLConfig()
    config.mobile.num_benchmark_runs = 5
    config.mobile.warmup_runs = 2
    config.mobile.quality_loss_tolerance = 0.05  # 5% max allowed drop
    return QuantizationEvaluator(config=config)


@pytest.fixture
def sample_data():
    torch.manual_seed(42)
    inputs = torch.randn(8, 32)
    targets = torch.randint(0, 10, (8,))
    return inputs, targets


def test_measure_variant_fp32(evaluator, sample_data):
    model = SimpleTestModel()
    inputs, targets = sample_data

    result = evaluator.measure_variant(
        model=model,
        sample_inputs=inputs,
        target_outputs=targets,
        variant_name="FP32 Test",
        precision="fp32",
        baseline_quality=None
    )

    assert isinstance(result, QuantizationVariantResult)
    assert result.precision == "fp32"
    assert result.p50_latency_ms > 0.0
    assert result.p95_latency_ms >= result.p50_latency_ms
    assert result.ram_mb > 0.0
    assert result.size_mb > 0.0
    assert result.energy_proxy > 0.0
    assert result.accepted is True
    assert result.rejection_reason is None


def test_quantize_fp16(evaluator, sample_data):
    model = SimpleTestModel()
    fp16_model = evaluator.quantize_fp16(model)
    inputs, targets = sample_data

    result = evaluator.measure_variant(
        model=fp16_model,
        sample_inputs=inputs,
        target_outputs=targets,
        variant_name="FP16 Test",
        precision="fp16",
        baseline_quality=1.0
    )

    assert result.precision == "fp16"
    assert result.p50_latency_ms > 0.0


def test_quantize_int8_dynamic(evaluator, sample_data):
    model = SimpleTestModel()
    int8_model = evaluator.quantize_int8_dynamic(model)
    inputs, targets = sample_data

    result = evaluator.measure_variant(
        model=int8_model,
        sample_inputs=inputs,
        target_outputs=targets,
        variant_name="INT8 Dynamic Test",
        precision="int8_dynamic",
        baseline_quality=1.0
    )

    assert result.precision == "int8_dynamic"
    assert result.p50_latency_ms > 0.0


def test_quality_loss_tolerance_rejection(evaluator, sample_data):
    model = SimpleTestModel()
    inputs, targets = sample_data

    def degraded_eval_fn(m, inp, tgt):
        # Force a bad quality score of 0.50 when baseline quality was 1.0
        return 0.50

    result = evaluator.measure_variant(
        model=model,
        sample_inputs=inputs,
        target_outputs=targets,
        variant_name="Degraded Variant",
        precision="int8_dynamic",
        baseline_quality=1.0,
        eval_fn=degraded_eval_fn,
        tolerance=0.05  # 5% max allowed drop
    )

    assert result.accepted is False
    assert result.quality_loss_pct == pytest.approx(50.0)
    assert result.rejection_reason is not None
    assert "exceeds predefined tolerance" in result.rejection_reason


def test_evaluate_all_variants(evaluator, sample_data):
    model = SimpleTestModel()
    inputs, targets = sample_data

    report = evaluator.evaluate_all_variants(
        fp32_model=model,
        sample_inputs=inputs,
        target_outputs=targets
    )

    assert isinstance(report, QuantizationReport)
    assert report.baseline.precision == "fp32"
    assert len(report.variants) == 3  # FP16, INT8 Dynamic, INT8 Static
    assert report.tolerance_pct == 5.0
    report_dict = report.to_dict()
    assert "baseline" in report_dict
    assert "variants" in report_dict
