"""Local-First Experiment Tracking System for Bi-ISL.

Provides local-first structured JSON/YAML experiment tracking for E0-E10 runs.
Captures run_id, experiment_id, timestamp, git commit, configuration,
dataset version, split hashes, seed, hardware info, training curves,
evaluation results, checkpoint paths, elapsed time, and errors/warnings.

Supports optional MLflow and Weights & Biases integrations via adapter hooks.
Includes run comparison tools for side-by-side metric comparison.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config import BiISLConfig
from src.utils.reproducibility import get_git_info, get_device_info


class BaseTrackerAdapter:
    """Optional adapter interface for external trackers (MLflow, WandB)."""

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


class ExperimentTracker:
    """Local-first experiment tracker for Bi-ISL runs."""

    def __init__(
        self,
        experiment_id: str = "E1",
        config: Optional[Union[BiISLConfig, Dict[str, Any]]] = None,
        base_dir: str = "./artifacts/runs",
        dataset_version: str = "1.0",
        split_hashes: Optional[Dict[str, str]] = None,
        external_adapter: Optional[BaseTrackerAdapter] = None,
    ):
        self.experiment_id = experiment_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.base_dir = Path(base_dir) / experiment_id / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(config, BiISLConfig):
            self.config_dict = config.to_dict()
            self.seed = config.training.seed
        elif isinstance(config, dict):
            self.config_dict = config
            self.seed = config.get("training", {}).get("seed", 42)
        else:
            default_cfg = BiISLConfig()
            self.config_dict = default_cfg.to_dict()
            self.seed = default_cfg.training.seed

        self.dataset_version = dataset_version
        self.split_hashes = split_hashes or {}
        self.external_adapter = external_adapter

        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.elapsed_time_seconds: float = 0.0

        self.training_curves: Dict[str, List[Dict[str, Any]]] = {}
        self.evaluation_results: Dict[str, Any] = {}
        self.checkpoint_paths: List[str] = []
        self.errors_and_warnings: List[Dict[str, Any]] = []

        self.git_info = get_git_info()
        self.hardware_info = get_device_info()

        if self.external_adapter:
            self.external_adapter.log_params(self.config_dict)

    def log_metric(self, metric_name: str, value: float, step: int) -> None:
        """Log a single metric step to training curves."""
        if metric_name not in self.training_curves:
            self.training_curves[metric_name] = []

        entry = {"step": step, "value": value, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.training_curves[metric_name].append(entry)

        if self.external_adapter:
            self.external_adapter.log_metric(metric_name, value, step)

    def log_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Log final evaluation metric results dictionary."""
        self.evaluation_results.update(results)
        if self.external_adapter:
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    self.external_adapter.log_metric(f"eval/{k}", float(v))

    def log_checkpoint(self, checkpoint_path: str) -> None:
        """Record path to a saved model checkpoint."""
        self.checkpoint_paths.append(str(checkpoint_path))

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.errors_and_warnings.append(
            {"type": "warning", "message": message, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.errors_and_warnings.append(
            {"type": "error", "message": message, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def finish(self, status: str = "completed") -> Dict[str, Any]:
        """Finalize run log, compute total execution time, and save run.json."""
        self.end_time = time.time()
        self.elapsed_time_seconds = round(self.end_time - self.start_time, 3)

        run_summary = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "status": status,
            "git": self.git_info,
            "configuration": self.config_dict,
            "dataset_version": self.dataset_version,
            "split_hashes": self.split_hashes,
            "seed": self.seed,
            "hardware": self.hardware_info,
            "training_curves": self.training_curves,
            "evaluation_results": self.evaluation_results,
            "checkpoint_paths": self.checkpoint_paths,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "errors_and_warnings": self.errors_and_warnings,
        }

        run_file = self.base_dir / "run.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, default=str)

        if self.external_adapter:
            self.external_adapter.finish()

        return run_summary


def load_run(run_dir: str) -> Dict[str, Any]:
    """Load run summary JSON from a run directory."""
    run_file = Path(run_dir) / "run.json"
    if not run_file.exists():
        raise FileNotFoundError(f"Run summary not found at {run_file}")
    with open(run_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_runs(
    runs_dir: str = "./artifacts/runs",
    experiment_id: Optional[str] = None,
    run_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search, filter, and compare recorded experiment runs."""
    base_path = Path(runs_dir)
    comparison_results = []

    if not base_path.exists():
        return comparison_results

    json_paths = list(base_path.glob("**/run.json"))

    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                run_data = json.load(f)

            if experiment_id and run_data.get("experiment_id") != experiment_id:
                continue

            if run_ids and run_data.get("run_id") not in run_ids:
                continue

            eval_res = run_data.get("evaluation_results", {})
            summary_entry = {
                "run_id": run_data.get("run_id"),
                "experiment_id": run_data.get("experiment_id"),
                "timestamp": run_data.get("timestamp"),
                "status": run_data.get("status"),
                "seed": run_data.get("seed"),
                "elapsed_time_s": run_data.get("elapsed_time_seconds"),
                "bleu4": eval_res.get("bleu4", eval_res.get("bleu_4")),
                "chrf": eval_res.get("chrf"),
                "usr": eval_res.get("usr", eval_res.get("unsupported_slot_rate")),
                "ece": eval_res.get("ece"),
                "p95_latency_ms": eval_res.get("p95_latency_ms"),
                "evaluation_results": eval_res,
                "path": str(jp.parent),
            }
            comparison_results.append(summary_entry)
        except Exception:
            pass

    comparison_results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return comparison_results


def format_comparison_table(comparison_results: List[Dict[str, Any]]) -> str:
    """Format comparison results into a clean markdown table."""
    if not comparison_results:
        return "No runs found for comparison."

    headers = ["Run ID", "Exp ID", "Status", "Seed", "BLEU-4", "chrF++", "USR", "p95 (ms)", "Time (s)"]
    rows = []
    for r in comparison_results:
        bleu = f"{r.get('bleu4'):.2f}" if r.get('bleu4') is not None else "N/A"
        chrf = f"{r.get('chrf'):.2f}" if r.get('chrf') is not None else "N/A"
        usr = f"{r.get('usr'):.4f}" if r.get('usr') is not None else "N/A"
        lat = f"{r.get('p95_latency_ms'):.1f}" if r.get('p95_latency_ms') is not None else "N/A"
        elapsed = f"{r.get('elapsed_time_s'):.1f}" if r.get('elapsed_time_s') is not None else "N/A"

        row = [
            str(r.get("run_id")),
            str(r.get("experiment_id")),
            str(r.get("status")),
            str(r.get("seed")),
            bleu,
            chrf,
            usr,
            lat,
            elapsed,
        ]
        rows.append(row)

    table_str = "| " + " | ".join(headers) + " |\n"
    table_str += "| " + " | ".join([":---"] * len(headers)) + " |\n"
    for r in rows:
        table_str += "| " + " | ".join(r) + " |\n"

    return table_str


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bi-ISL Experiment Run Comparator")
    parser.add_argument("--runs-dir", type=str, default="./artifacts/runs", help="Directory containing experiment run logs")
    parser.add_argument("--experiment-id", type=str, default=None, help="Filter by experiment ID (e.g. E1)")
    args = parser.parse_args()

    results = compare_runs(runs_dir=args.runs_dir, experiment_id=args.experiment_id)
    print(format_comparison_table(results))
