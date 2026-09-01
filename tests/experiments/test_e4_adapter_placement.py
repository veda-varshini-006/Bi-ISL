"""Unit tests for E4 Adapter Placement Experiment (Prompt 41)."""

import os
import tempfile
import pytest

from src.experiments.e4_adapter_placement import run_adapter_placement_experiment


def test_run_adapter_placement_experiment():
    """Test evaluating all 4 candidate placements on validation set and generating report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, doc_p, summary = run_adapter_placement_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(doc_p)

        assert len(summary["placements_evaluated"]) == 4
        assert summary["best_placement"] in summary["placements_evaluated"]
        assert summary["best_trainable_parameters"] < 50000
