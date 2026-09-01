"""Smoke tests for pipeline integration."""

import pytest
from src.utils.config import BiISLConfig
from src.utils.reproducibility import capture_environment_metadata
from src.utils.experiment_tracker import ExperimentTracker

def test_full_pipeline_smoke(tmp_path):
    config = BiISLConfig()
    tracker = ExperimentTracker(experiment_id="E0", config=config, base_dir=str(tmp_path))
    
    metadata = capture_environment_metadata(seed=config.training.seed, model_config=config.to_dict())
    assert "git" in metadata
    
    tracker.log_metric("loss", 0.1, step=1)
    tracker.log_evaluation_results({"bleu4": 30.0})
    summary = tracker.finish()

    assert summary["status"] == "completed"
    assert summary["evaluation_results"]["bleu4"] == 30.0
