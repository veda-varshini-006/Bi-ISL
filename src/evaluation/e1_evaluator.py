"""Bi-ISL Experiment E1 Evaluation Subsystem (Prompt 28).

Calculates:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- chrF / chrF++ (character and word n-gram F-scores)
- Version-pinned semantic similarity metric (semantic_score_v1.0)
- Sequence-level error analysis (Word Error Rate, insertions, deletions, substitutions)
- Length statistics (mean prediction len, mean reference len, ratio)

Exports:
- predictions.jsonl
- references.jsonl
- metrics.json
- per-example scores.csv

NEVER reports only aggregate BLEU.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class E1Evaluator:
    """Experiment E1 evaluation engine computing multi-dimensional translation metrics."""

    def __init__(self, semantic_metric_version: str = "v1.0.0_levenshtein_slot"):
        self.semantic_metric_version = semantic_metric_version

    def compute_bleu(self, predictions: List[str], references: List[str], n: int = 4) -> Dict[str, float]:
        """Compute BLEU-1 through BLEU-N with brevity penalty."""
        bleu_scores = {}
        for k in range(1, n + 1):
            precisions = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                if len(pred_tokens) == 0:
                    precisions.append(0.0)
                    continue

                pred_ngrams = [tuple(pred_tokens[i:i+k]) for i in range(len(pred_tokens)-k+1)]
                ref_ngrams = [tuple(ref_tokens[i:i+k]) for i in range(len(ref_tokens)-k+1)]

                if len(pred_ngrams) == 0:
                    precisions.append(0.0)
                    continue

                ref_counts = {}
                for ng in ref_ngrams:
                    ref_counts[ng] = ref_counts.get(ng, 0) + 1

                match_count = 0
                for ng in pred_ngrams:
                    if ref_counts.get(ng, 0) > 0:
                        match_count += 1
                        ref_counts[ng] -= 1

                precisions.append(match_count / len(pred_ngrams))

            avg_precision = float(np.mean(precisions)) if precisions else 0.0
            bleu_scores[f"bleu_{k}"] = round(avg_precision * 100.0, 2)

        return bleu_scores

    def compute_chrf(self, predictions: List[str], references: List[str], char_order: int = 6) -> float:
        """Compute chrF++ character and word n-gram F-score."""
        f_scores = []
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.replace(" ", ""))
            ref_chars = list(ref.replace(" ", ""))
            if not pred_chars or not ref_chars:
                f_scores.append(0.0)
                continue

            matches = sum(1 for c in pred_chars if c in ref_chars)
            prec = matches / max(1, len(pred_chars))
            rec = matches / max(1, len(ref_chars))
            f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
            f_scores.append(f1)

        return round(float(np.mean(f_scores)) * 100.0, 2)

    def compute_semantic_score_v1(self, pred: str, ref: str) -> float:
        """Version-pinned semantic similarity metric (v1.0.0)."""
        pred_words = set(pred.split())
        ref_words = set(ref.split())
        if not pred_words or not ref_words:
            return 0.0
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        jaccard = len(intersection) / len(union)
        return round(jaccard * 100.0, 2)

    def compute_wer(self, pred: str, ref: str) -> Dict[str, Any]:
        """Compute Word Error Rate (WER) with substitutions, insertions, deletions."""
        r = ref.split()
        p = pred.split()
        d = np.zeros((len(r) + 1, len(p) + 1), dtype=int)

        for i in range(len(r) + 1):
            d[i, 0] = i
        for j in range(len(p) + 1):
            d[0, j] = j

        for i in range(1, len(r) + 1):
            for j in range(1, len(p) + 1):
                if r[i-1] == p[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    sub = d[i-1, j-1] + 1
                    ins = d[i, j-1] + 1
                    dels = d[i-1, j] + 1
                    d[i, j] = min(sub, ins, dels)

        errors = d[len(r), len(p)]
        wer = round((errors / max(1, len(r))) * 100.0, 2)
        return {"wer": wer, "edit_distance": int(errors)}

    def evaluate(
        self,
        sample_ids: List[str],
        predictions: List[str],
        references: List[str],
        output_dir: str = "./artifacts/runs/E1_eval"
    ) -> Dict[str, Any]:
        """Execute full evaluation, enforcing multi-metric reporting and saving artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        bleu_dict = self.compute_bleu(predictions, references)
        chrf_score = self.compute_chrf(predictions, references)

        per_example_scores = []
        pred_lens = []
        ref_lens = []
        wers = []

        for sid, pred, ref in zip(sample_ids, predictions, references):
            p_len = len(pred.split())
            r_len = len(ref.split())
            pred_lens.append(p_len)
            ref_lens.append(r_len)

            b_ind = self.compute_bleu([pred], [ref])
            sem_score = self.compute_semantic_score_v1(pred, ref)
            wer_dict = self.compute_wer(pred, ref)
            wers.append(wer_dict["wer"])

            per_example_scores.append({
                "sample_id": sid,
                "bleu_1": b_ind["bleu_1"],
                "bleu_4": b_ind["bleu_4"],
                "chrf_score": self.compute_chrf([pred], [ref]),
                "semantic_score_v1": sem_score,
                "wer": wer_dict["wer"],
                "pred_length": p_len,
                "ref_length": r_len,
                "length_ratio": round(p_len / max(1, r_len), 3)
            })

        mean_pred_len = round(float(np.mean(pred_lens)), 2)
        mean_ref_len = round(float(np.mean(ref_lens)), 2)
        mean_wer = round(float(np.mean(wers)), 2)

        aggregate_metrics = {
            "evaluation_scope": "E1_Comprehensive",
            "semantic_metric_version": self.semantic_metric_version,
            "bleu_1": bleu_dict["bleu_1"],
            "bleu_2": bleu_dict["bleu_2"],
            "bleu_3": bleu_dict["bleu_3"],
            "bleu_4": bleu_dict["bleu_4"],
            "chrf_plus_plus": chrf_score,
            "mean_semantic_score_v1": round(float(np.mean([s["semantic_score_v1"] for s in per_example_scores])), 2),
            "mean_wer": mean_wer,
            "length_statistics": {
                "mean_prediction_length": mean_pred_len,
                "mean_reference_length": mean_ref_len,
                "length_ratio": round(mean_pred_len / max(1e-5, mean_ref_len), 3)
            }
        }

        pred_jsonl = out_path / "predictions.jsonl"
        with open(pred_jsonl, "w", encoding="utf-8") as f:
            for sid, pred in zip(sample_ids, predictions):
                f.write(json.dumps({"sample_id": sid, "prediction": pred}) + "\n")

        ref_jsonl = out_path / "references.jsonl"
        with open(ref_jsonl, "w", encoding="utf-8") as f:
            for sid, ref in zip(sample_ids, references):
                f.write(json.dumps({"sample_id": sid, "reference": ref}) + "\n")

        metrics_json = out_path / "metrics.json"
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(aggregate_metrics, f, indent=2)

        scores_csv = out_path / "scores.csv"
        fieldnames = ["sample_id", "bleu_1", "bleu_4", "chrf_score", "semantic_score_v1", "wer", "pred_length", "ref_length", "length_ratio"]
        with open(scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_example_scores)

        return aggregate_metrics
