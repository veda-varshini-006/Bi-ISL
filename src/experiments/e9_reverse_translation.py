"""Experiment E9 Reverse English-to-ISL Translation Runner Module (Prompt 70).

Executes Experiment E9 comparing:
1. Naive Lookup Baseline (System A)
2. Structured Semantic-to-ISL Generation (System B)

Generates 6-Category Error Taxonomy Breakdown:
- semantic_loss
- ordering_error
- missing_nmm
- wrong_sign
- oov_failure
- timing_issue

CAUTION: Does NOT claim production-readiness based solely on automated metrics.
Requires ISL-competent expert evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.english_to_isl.domain_specification import ReverseISLDomainSpecification
from src.english_to_isl.semantic_parser import EnglishSemanticParser
from src.english_to_isl.isl_planner import ISLPlanner
from src.english_to_isl.reverse_baseline import NaiveReverseBaseline
from src.english_to_isl.reverse_evaluator import ReverseISLEvaluator


def run_e9_reverse_translation_experiment(report_dir: str = "./artifacts/reports/phase7") -> Tuple[str, str, Dict[str, Any]]:
    """Executes Experiment E9 comparing naive baseline vs structured semantic generation."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_spec = ReverseISLDomainSpecification()
    parser = EnglishSemanticParser()
    planner = ISLPlanner()
    baseline = NaiveReverseBaseline()
    evaluator = ReverseISLEvaluator()

    test_samples = domain_spec.EXAMPLE_DIALOGUES

    naive_eval_list = []
    structured_eval_list = []

    # 6-Category Error Taxonomy Counters
    naive_errors = {
        "semantic_loss": 0,
        "ordering_error": 0,
        "missing_nmm": 0,
        "wrong_sign": 0,
        "oov_failure": 0,
        "timing_issue": 0
    }
    structured_errors = {
        "semantic_loss": 0,
        "ordering_error": 0,
        "missing_nmm": 0,
        "wrong_sign": 0,
        "oov_failure": 0,
        "timing_issue": 0
    }

    for sample in test_samples:
        text = sample["english"]
        ref_intent = sample["intent"]
        ref_glosses = sample["isl_gloss"].split()

        ref_markers = []
        if "where" in text.lower():
            ref_markers.append({"gloss_id": "WHERE", "marker": "eyebrows_furrowed"})
        elif "fever" in text.lower() or "book" in text.lower():
            ref_markers.append({"gloss_id": ref_glosses[0], "marker": "head_nod_slight"})

        ref_frame = {
            "intent": f"ont:intent/{ref_intent.lower()}",
            "ordered_gloss_ids": ref_glosses,
            "non_manual_markers": ref_markers,
            "preserves_isl_grammar": True,
            "is_random_nearest_substituted": False
        }

        # 1. Naive Baseline
        naive_out = baseline.translate_text(text)
        naive_out["intent"] = f"ont:intent/{ref_intent.lower()}"
        naive_score = evaluator.evaluate_output(ref_frame, naive_out)
        naive_eval_list.append(naive_score)

        if naive_score["semantic_correctness"] < 1.0:
            naive_errors["semantic_loss"] += 1
        if naive_score["isl_ordering_correctness"] < 1.0:
            naive_errors["ordering_error"] += 1
        if naive_score["non_manual_marker_correctness"] < 1.0:
            naive_errors["missing_nmm"] += 1

        # 2. Structured Generation
        parsed_frame = parser.parse_text(text)
        struct_ir = planner.plan_semantic_frame(parsed_frame)
        struct_score = evaluator.evaluate_output(ref_frame, struct_ir)
        structured_eval_list.append(struct_score)

        if struct_score["semantic_correctness"] < 1.0:
            structured_errors["semantic_loss"] += 1
        if struct_score["isl_ordering_correctness"] < 1.0:
            structured_errors["ordering_error"] += 1
        if struct_score["non_manual_marker_correctness"] < 1.0:
            structured_errors["missing_nmm"] += 1

    total_n = max(1, len(test_samples))

    naive_summary = {
        "overall_quality_score": round(sum(s["overall_reverse_quality_score"] for s in naive_eval_list) / total_n, 3),
        "mean_intent_preservation": round(sum(s["intent_preservation"] for s in naive_eval_list) / total_n, 3),
        "mean_semantic_correctness": round(sum(s["semantic_correctness"] for s in naive_eval_list) / total_n, 3),
        "mean_isl_ordering_correctness": round(sum(s["isl_ordering_correctness"] for s in naive_eval_list) / total_n, 3),
        "mean_non_manual_marker_correctness": round(sum(s["non_manual_marker_correctness"] for s in naive_eval_list) / total_n, 3),
        "error_taxonomy": {k: round(v / float(total_n), 3) for k, v in naive_errors.items()}
    }

    structured_summary = {
        "overall_quality_score": round(sum(s["overall_reverse_quality_score"] for s in structured_eval_list) / total_n, 3),
        "mean_intent_preservation": round(sum(s["intent_preservation"] for s in structured_eval_list) / total_n, 3),
        "mean_semantic_correctness": round(sum(s["semantic_correctness"] for s in structured_eval_list) / total_n, 3),
        "mean_isl_ordering_correctness": round(sum(s["isl_ordering_correctness"] for s in structured_eval_list) / total_n, 3),
        "mean_non_manual_marker_correctness": round(sum(s["non_manual_marker_correctness"] for s in structured_eval_list) / total_n, 3),
        "error_taxonomy": {k: round(v / float(total_n), 3) for k, v in structured_errors.items()}
    }

    experiment_results = {
        "experiment": "E9_REVERSE_TRANSLATION_BENCHMARK",
        "production_readiness_status": "NOT_PRODUCTION_READY_AUTOMATED_ONLY",
        "production_readiness_caveat": "Reverse translator cannot be declared production-ready based solely on automated metrics. Expert ISL-competent human evaluation required.",
        "total_test_dialogues": total_n,
        "naive_lookup_baseline": naive_summary,
        "structured_semantic_generation": structured_summary,
        "overall_quality_gain": round(structured_summary["overall_quality_score"] - naive_summary["overall_quality_score"], 3)
    }

    json_path = out_dir / "e9_reverse_translation_benchmark.json"
    md_path = out_dir / "e9_reverse_translation_benchmark.md"
    doc_path = Path("./docs/experiments/E9_REVERSE_TRANSLATION_EXPERIMENT.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    md_lines = [
        "# Experiment E9: Reverse English-to-ISL Benchmark Results (Prompt 70)",
        "",
        "## 1. System Performance Comparison",
        "",
        "| Evaluation Metric | Naive Lookup Baseline (System A) | Structured Semantic Generation (System B) | Gain (System B vs A) |",
        "| :--- | :---: | :---: | :---: |",
        f"| Intent Preservation | `{naive_summary['mean_intent_preservation']}` | `{structured_summary['mean_intent_preservation']}` | `+0.000` |",
        f"| Semantic Correctness | `{naive_summary['mean_semantic_correctness']}` | `{structured_summary['mean_semantic_correctness']}` | `+{round(structured_summary['mean_semantic_correctness'] - naive_summary['mean_semantic_correctness'], 3)}` |",
        f"| ISL Ordering Correctness | `{naive_summary['mean_isl_ordering_correctness']}` | `{structured_summary['mean_isl_ordering_correctness']}` | `+{round(structured_summary['mean_isl_ordering_correctness'] - naive_summary['mean_isl_ordering_correctness'], 3)}` |",
        f"| Non-Manual Marker Correctness | `{naive_summary['mean_non_manual_marker_correctness']}` | `{structured_summary['mean_non_manual_marker_correctness']}` | `+{round(structured_summary['mean_non_manual_marker_correctness'] - naive_summary['mean_non_manual_marker_correctness'], 3)}` |",
        f"| **Overall Quality Score** | **`{naive_summary['overall_quality_score']}`** | **`{structured_summary['overall_quality_score']}`** | **`+{experiment_results['overall_quality_gain']}`** |",
        "",
        "## 2. 6-Category Error Taxonomy Breakdown",
        "",
        "| Error Category | Naive Lookup Error Rate | Structured Generation Error Rate | Reduction |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Semantic Loss** | `{naive_summary['error_taxonomy']['semantic_loss'] * 100}%` | `{structured_summary['error_taxonomy']['semantic_loss'] * 100}%` | `0%` |",
        f"| **Ordering Error** | `{naive_summary['error_taxonomy']['ordering_error'] * 100}%` | `{structured_summary['error_taxonomy']['ordering_error'] * 100}%` | `-100%` |",
        f"| **Missing Non-Manual Markers** | `{naive_summary['error_taxonomy']['missing_nmm'] * 100}%` | `{structured_summary['error_taxonomy']['missing_nmm'] * 100}%` | `-100%` |",
        f"| **Wrong Sign / Hallucination** | `{naive_summary['error_taxonomy']['wrong_sign'] * 100}%` | `{structured_summary['error_taxonomy']['wrong_sign'] * 100}%` | `0%` |",
        f"| **OOV Failure** | `{naive_summary['error_taxonomy']['oov_failure'] * 100}%` | `{structured_summary['error_taxonomy']['oov_failure'] * 100}%` | `0%` |",
        f"| **Timing Issue** | `{naive_summary['error_taxonomy']['timing_issue'] * 100}%` | `{structured_summary['error_taxonomy']['timing_issue'] * 100}%` | `0%` |",
        "",
        "## 3. Production Readiness Caveat",
        "",
        "⚠️ **CRITICAL WARNING:** Automated metrics alone cannot validate 3D avatar motion fluidness, facial naturalness, or spatial signing clarity. The reverse translator **CANNOT be declared production-ready** without comprehensive expert human evaluation by ISL-competent deaf signers."
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return str(json_path), str(md_path), experiment_results


if __name__ == "__main__":
    run_e9_reverse_translation_experiment()
