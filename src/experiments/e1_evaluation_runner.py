"""Experiment Runner: Experiment E1 Multi-Metric Evaluation (Prompt 28)."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.evaluation.e1_evaluator import E1Evaluator


def run_e1_evaluation_experiment(
    sample_ids: Optional[List[str]] = None,
    predictions: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
    output_dir: str = "./artifacts/runs/E1_eval"
) -> Tuple[str, Dict[str, Any]]:
    """Run full E1 multi-metric evaluation and export all required artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if sample_ids is None:
        sample_ids = [f"e1_sample_{i}" for i in range(5)]
    if predictions is None:
        predictions = [
            "hello world sign language",
            "good morning everyone",
            "thank you very much",
            "please help me today",
            "nice to meet you"
        ]
    if references is None:
        references = [
            "hello world sign language",
            "good morning everyone",
            "thank you so much",
            "please help me",
            "nice to meet you"
        ]

    evaluator = E1Evaluator(semantic_metric_version="v1.0.0_levenshtein_slot")
    metrics = evaluator.evaluate(sample_ids, predictions, references, output_dir=output_dir)

    required_keys = ["bleu_1", "bleu_4", "chrf_plus_plus", "mean_semantic_score_v1", "mean_wer", "length_statistics"]
    for k in required_keys:
        if k not in metrics:
            raise ValueError(f"E1 Evaluation Rule Violation: Missing required metric '{k}'!")

    return output_dir, metrics


if __name__ == "__main__":
    run_e1_evaluation_experiment()
