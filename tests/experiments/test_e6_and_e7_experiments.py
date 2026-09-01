"""Unit tests for E6 Joint Mechanism & E7 Generalization Experiments (Prompts 59 & 60)."""

import os
import tempfile
import pytest

from src.experiments.e6_joint_mechanism import run_e6_joint_experiment
from src.experiments.e7_generalization import run_e7_generalization_experiment


def test_run_e6_joint_experiment():
    """Test running E6 joint mechanism multi-seed experiment with paired statistical testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, report = run_e6_joint_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        assert len(report["seeds_evaluated"]) == 5
        stats = report["statistical_analysis"]
        assert stats["relationship_classification"] == "SUPER_ADDITIVE_SYNERGY"
        assert stats["p_value"] < 0.01


def test_run_e7_generalization_experiment():
    """Test running E7 generalization benchmark across 6 operational regimes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p, md_p, report = run_e7_generalization_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        regimes = report["operational_regimes"]
        assert len(regimes) == 6
        assert "SEEN_SIGNER_CLEAN" in regimes
        assert "CROSS_DOMAIN_INCLUDE" in regimes


def test_documentation_file_exists():
    """Verify E7_GENERALIZATION_BENCHMARK.md exists."""
    doc_path = "./docs/experiments/E7_GENERALIZATION_BENCHMARK.md"
    assert os.path.exists(doc_path)
