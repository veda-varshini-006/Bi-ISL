"""Integration tests for Experiment E8: Mobile Edge Benchmark Experiment (Prompt 88)."""

import os
import pytest
from src.experiments.e8_mobile_benchmark import E8MobileBenchmarkExperiment
from src.utils.config import BiISLConfig


def test_e8_mobile_benchmark_experiment_execution(tmp_path):
    config = BiISLConfig()
    config.mobile.num_benchmark_runs = 5
    config.mobile.warmup_runs = 2
    config.experiment.output_dir = str(tmp_path)

    exp = E8MobileBenchmarkExperiment(config=config)
    exp.setup()

    assert exp.model is not None
    assert exp.evaluator is not None
    assert exp.sample_inputs is not None

    results = exp.run()

    assert isinstance(results, dict)
    assert "baseline" in results
    assert "variants" in results
    assert len(results["variants"]) == 3

    # Check generated JSON report file
    report_file = os.path.join(str(tmp_path), "quantization_report.json")
    assert os.path.exists(report_file)
    assert os.path.getsize(report_file) > 0

    exp.teardown()
    assert exp.model is None
