"""Unit tests for the Bi-ISL Local-First Experiment Tracking System."""

import json
import os
import tempfile
import pytest

from src.utils.config import BiISLConfig
from src.utils.experiment_tracker import (
    ExperimentTracker,
    load_run,
    compare_runs,
    format_comparison_table,
    BaseTrackerAdapter
)


def test_experiment_tracker_lifecycle():
    """Test full tracking lifecycle: init, log metrics/eval, checkpoint, finish."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BiISLConfig()
        config.training.seed = 1234
        split_hashes = {"train": "hash_abc", "test": "hash_xyz"}

        tracker = ExperimentTracker(
            experiment_id="E1",
            config=config,
            base_dir=tmp_dir,
            dataset_version="1.0",
            split_hashes=split_hashes
        )

        # Log training curves
        tracker.log_metric("loss", 0.85, step=1)
        tracker.log_metric("loss", 0.42, step=2)
        tracker.log_metric("bleu", 12.5, step=1)
        tracker.log_metric("bleu", 24.8, step=2)

        # Log checkpoints, warnings, errors
        tracker.log_checkpoint("/path/to/model_epoch_1.pt")
        tracker.log_checkpoint("/path/to/model_epoch_2.pt")
        tracker.log_warning("High GPU temperature warning")
        tracker.log_error("Sample #42 missing keypoint data")

        # Log final evaluation results
        eval_metrics = {
            "bleu4": 28.5,
            "chrf": 45.2,
            "usr": 0.021,
            "ece": 0.045,
            "p95_latency_ms": 145.0
        }
        tracker.log_evaluation_results(eval_metrics)

        # Finish run
        summary = tracker.finish(status="completed")

        # Verify output summary structure
        assert summary["experiment_id"] == "E1"
        assert summary["seed"] == 1234
        assert summary["dataset_version"] == "1.0"
        assert summary["split_hashes"] == split_hashes
        assert len(summary["checkpoint_paths"]) == 2
        assert len(summary["errors_and_warnings"]) == 2
        assert summary["evaluation_results"]["bleu4"] == 28.5
        assert summary["elapsed_time_seconds"] >= 0.0

        # Verify run.json file on disk
        run_json_path = os.path.join(tmp_dir, "E1", tracker.run_id, "run.json")
        assert os.path.exists(run_json_path)

        loaded_summary = load_run(os.path.join(tmp_dir, "E1", tracker.run_id))
        assert loaded_summary["run_id"] == tracker.run_id
        assert loaded_summary["evaluation_results"]["chrf"] == 45.2


def test_compare_runs_and_formatting():
    """Test comparing multiple recorded runs and table formatting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create Run 1
        t1 = ExperimentTracker(experiment_id="E2", base_dir=tmp_dir)
        t1.log_evaluation_results({"bleu4": 22.0, "chrf": 40.0, "usr": 0.05, "p95_latency_ms": 180.0})
        r1_id = t1.run_id
        t1.finish()

        # Create Run 2
        t2 = ExperimentTracker(experiment_id="E2", base_dir=tmp_dir)
        t2.log_evaluation_results({"bleu4": 27.5, "chrf": 48.0, "usr": 0.01, "p95_latency_ms": 150.0})
        r2_id = t2.run_id
        t2.finish()

        # Compare runs for experiment E2
        results = compare_runs(runs_dir=tmp_dir, experiment_id="E2")
        assert len(results) == 2

        # Format table
        table_output = format_comparison_table(results)
        assert "E2" in table_output
        assert r1_id in table_output
        assert r2_id in table_output
        assert "27.50" in table_output or "22.00" in table_output


def test_external_adapter_mock():
    """Test optional external adapter interface hook."""
    class MockAdapter(BaseTrackerAdapter):
        def __init__(self):
            self.logged_metrics = {}
            self.logged_params = {}
            self.finished = False

        def log_metric(self, name: str, value: float, step: int = None):
            self.logged_metrics[name] = value

        def log_params(self, params: dict):
            self.logged_params.update(params)

        def finish(self):
            self.finished = True

    mock_adapter = MockAdapter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(experiment_id="E3", base_dir=tmp_dir, external_adapter=mock_adapter)
        tracker.log_metric("loss", 0.5, step=1)
        tracker.log_evaluation_results({"accuracy": 0.95})
        tracker.finish()

    assert mock_adapter.logged_metrics["loss"] == 0.5
    assert mock_adapter.logged_metrics["eval/accuracy"] == 0.95
    assert mock_adapter.finished is True
