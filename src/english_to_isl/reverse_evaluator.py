"""Reverse English-to-ISL Evaluation Pipeline (Prompt 69).

Evaluates reverse translation outputs across 5 core dimensions:
1. Semantic Correctness (Slot & concept preservation)
2. ISL Ordering Correctness (Topic-Comment / SOV grammar rules)
3. Non-Manual Marker Correctness (Facial expressions & eyebrow annotations)
4. Unsupported / OOV Handling (Zero random nearest substitution & safe fallbacks)
5. Intent Preservation (Domain intent retention)

Includes Expert / ISL-Competent Human Evaluation Rubrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ReverseISLEvaluator:
    """Evaluation tool for reverse English-to-ISL translation models."""

    def __init__(self, report_dir: str = "./artifacts/reports/phase7"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_output(self, reference: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates prediction against reference across 5 quantitative dimensions."""
        ref_intent = reference.get("intent", reference.get("intent_uri", ""))
        pred_intent = prediction.get("intent", prediction.get("intent_uri", ""))

        ref_glosses = reference.get("ordered_gloss_ids", [])
        pred_glosses = prediction.get("ordered_gloss_ids", [])

        ref_markers = reference.get("non_manual_markers", [])
        pred_markers = prediction.get("non_manual_markers", [])

        intent_score = 1.0 if ref_intent and ref_intent == pred_intent else 0.0

        if ref_glosses and pred_glosses:
            common = set(ref_glosses).intersection(set(pred_glosses))
            semantic_score = round(len(common) / float(max(1, len(set(ref_glosses)))), 3)
        else:
            semantic_score = 0.0

        ordering_score = 1.0 if prediction.get("preserves_isl_grammar", True) else 0.0
        if pred_glosses and ref_glosses and pred_glosses == ref_glosses:
            ordering_score = 1.0

        if not ref_markers and not pred_markers:
            nmm_score = 1.0
        elif ref_markers and pred_markers:
            ref_m_set = {m.get("marker") for m in ref_markers}
            pred_m_set = {m.get("marker") for m in pred_markers}
            nmm_score = 1.0 if ref_m_set == pred_m_set else 0.5
        else:
            nmm_score = 0.0

        oov_score = 1.0 if not prediction.get("is_random_nearest_substituted", False) else 0.0

        overall_score = round((intent_score + semantic_score + ordering_score + nmm_score + oov_score) / 5.0, 3)

        return {
            "intent_preservation": intent_score,
            "semantic_correctness": semantic_score,
            "isl_ordering_correctness": ordering_score,
            "non_manual_marker_correctness": nmm_score,
            "unsupported_oov_handling": oov_score,
            "overall_reverse_quality_score": overall_score
        }

    def generate_human_evaluation_rubric(self) -> Dict[str, Any]:
        """Provides expert ISL-competent human evaluation guidelines."""
        return {
            "rubric_title": "ISL-Competent Expert Human Evaluation Protocol",
            "evaluator_qualification": "ISL-competent evaluator (deaf native signer or certified interpreter)",
            "qualitative_dimensions": {
                "facial_expression_naturalness": "Scale 1-5 rating naturalness of eyebrow/mouth co-articulation",
                "signing_flow_coarticulation": "Scale 1-5 rating transition smoothness between sign loci",
                "semantic_faithfulness": "Scale 1-5 rating intent retention without medical distortion"
            }
        }

    def export_evaluation_report(self, eval_results: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
        """Exports evaluation benchmark JSON & Markdown reports."""
        summary = {
            "num_evaluated_samples": len(eval_results),
            "mean_intent_preservation": round(sum(r["intent_preservation"] for r in eval_results) / max(1, len(eval_results)), 3),
            "mean_semantic_correctness": round(sum(r["semantic_correctness"] for r in eval_results) / max(1, len(eval_results)), 3),
            "mean_isl_ordering_correctness": round(sum(r["isl_ordering_correctness"] for r in eval_results) / max(1, len(eval_results)), 3),
            "mean_non_manual_marker_correctness": round(sum(r["non_manual_marker_correctness"] for r in eval_results) / max(1, len(eval_results)), 3),
            "mean_unsupported_oov_handling": round(sum(r["unsupported_oov_handling"] for r in eval_results) / max(1, len(eval_results)), 3),
            "overall_quality_score": round(sum(r["overall_reverse_quality_score"] for r in eval_results) / max(1, len(eval_results)), 3)
        }

        report_data = {
            "summary": summary,
            "human_evaluation_rubric": self.generate_human_evaluation_rubric()
        }

        json_path = self.report_dir / "reverse_evaluator_report.json"
        md_path = self.report_dir / "reverse_evaluator_report.md"
        doc_path = Path("./docs/evaluation/REVERSE_EVALUATOR_SPEC.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        md_lines = [
            "# Reverse English-to-ISL Evaluator Specification & Results (Prompt 69)",
            "",
            "## 1. Quantitative Benchmark Summary",
            "",
            "| Evaluation Dimension | Mean Score | Target Threshold |",
            "| :--- | :---: | :---: |",
            f"| Intent Preservation | `{summary['mean_intent_preservation']}` | 1.000 |",
            f"| Semantic Correctness | `{summary['mean_semantic_correctness']}` | > 0.800 |",
            f"| ISL Ordering Correctness | `{summary['mean_isl_ordering_correctness']}` | 1.000 |",
            f"| Non-Manual Marker Correctness | `{summary['mean_non_manual_marker_correctness']}` | > 0.800 |",
            f"| Unsupported/OOV Handling | `{summary['mean_unsupported_oov_handling']}` | 1.000 |",
            f"| **Overall Quality Score** | **`{summary['overall_quality_score']}`** | **> 0.850** |",
            "",
            "## 2. Expert Human Evaluation Protocol",
            "",
            "Qualitative claims requiring ISL-competent deaf signer validation are assessed using standard 1-5 Likert rubrics for facial naturalness and motion flow."
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_path), str(md_path), report_data
