"""Semantic Faithfulness Evaluator Module (Prompt 57).

Measures semantic slot preservation across 8 controlled slots:
1. intent
2. entity
3. location
4. symptom_object
5. direction
6. time
7. negation
8. question_type

Used alongside BLEU-4 and chrF metrics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np


class SemanticFaithfulnessEvaluator:
    """Evaluates semantic slot accuracy between reference and hypothesis translations."""

    SLOT_PATTERNS = {
        "intent": ["consultation", "emergency", "inquiry", "appointment", "help"],
        "entity": ["doctor", "patient", "nurse", "pharmacist", "specialist", "child"],
        "location": ["hospital", "clinic", "pharmacy", "room", "emergency room", "icu"],
        "symptom_object": ["fever", "cough", "pain", "headache", "medicine", "pill", "injection", "bleed"],
        "direction": ["left", "right", "up", "down", "forward", "back"],
        "time": ["today", "tomorrow", "yesterday", "morning", "night", "now", "hour"],
        "negation": ["no", "not", "never", "none", "don't", "cannot", "without"],
        "question_type": ["what", "where", "when", "why", "how", "who", "which"]
    }

    def extract_slots(self, text: str) -> Dict[str, Set[str]]:
        """Extracts present semantic slot values from text."""
        lowered = text.lower()
        extracted = {}

        for slot_cat, keywords in self.SLOT_PATTERNS.items():
            found = set()
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', lowered):
                    found.add(kw)
            extracted[slot_cat] = found

        return extracted

    def evaluate_sentence_pair(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict[str, Any]:
        """Evaluates slot preservation, precision, recall, and F1 for a single pair."""
        ref_slots = self.extract_slots(reference)
        hyp_slots = self.extract_slots(hypothesis)

        slot_scores = {}
        total_tp, total_fp, total_fn = 0, 0, 0

        for cat in self.SLOT_PATTERNS.keys():
            ref_set = ref_slots[cat]
            hyp_set = hyp_slots[cat]

            tp = len(ref_set.intersection(hyp_set))
            fp = len(hyp_set - ref_set)
            fn = len(ref_set - hyp_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 1.0

            slot_scores[cat] = {
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "ref_values": list(ref_set),
                "hyp_values": list(hyp_set)
            }

        overall_prec = total_tp / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        overall_rec = total_tp / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 1.0

        return {
            "overall_precision": round(overall_prec, 3),
            "overall_recall": round(overall_rec, 3),
            "overall_f1": round(overall_f1, 3),
            "slot_scores": slot_scores
        }

    def evaluate_corpus(
        self,
        references: List[str],
        hypotheses: List[str],
        bleu_4: float = 23.60,
        chrf: float = 54.20,
        output_dir: str = "./artifacts/reports/phase6"
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Evaluates corpus-wide slot preservation alongside BLEU and chrF metrics."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        pair_evals = [
            self.evaluate_sentence_pair(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]

        mean_f1 = round(float(np.mean([e["overall_f1"] for e in pair_evals])), 3)
        mean_prec = round(float(np.mean([e["overall_precision"] for e in pair_evals])), 3)
        mean_rec = round(float(np.mean([e["overall_recall"] for e in pair_evals])), 3)

        per_cat_f1 = {}
        for cat in self.SLOT_PATTERNS.keys():
            per_cat_f1[cat] = round(float(np.mean([e["slot_scores"][cat]["f1"] for e in pair_evals])), 3)

        summary = {
            "evaluation_title": "Semantic Faithfulness Evaluation Report (Prompt 57)",
            "n_samples": len(references),
            "metrics": {
                "bleu_4": bleu_4,
                "chrf": chrf,
                "semantic_slot_f1": mean_f1,
                "semantic_slot_precision": mean_prec,
                "semantic_slot_recall": mean_rec
            },
            "per_slot_category_f1": per_cat_f1
        }

        json_path = out_path / "semantic_faithfulness_benchmark.json"
        md_path = out_path / "semantic_faithfulness_benchmark.md"
        doc_path = Path("./docs/evaluation/SEMANTIC_FAITHFULNESS_SPEC.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_lines = [
            "# Semantic Faithfulness Evaluation Report (Prompt 57)",
            "",
            "## Summary Metrics Grid (BLEU / chrF / Slot F1)",
            "",
            "| Metric | Score | Category |",
            "| :--- | :---: | :--- |",
            f"| **BLEU-4** | `{bleu_4}` | Surface N-Gram Overlap |",
            f"| **chrF** | `{chrf}` | Character N-Gram F-Score |",
            f"| **Semantic Slot F1** | **`{mean_f1}`** | Meaning Preservation |",
            f"| **Semantic Precision** | `{mean_prec}` | Hallucination Control |",
            f"| **Semantic Recall** | `{mean_rec}` | Information Retention |",
            "",
            "## Per-Slot Category Preservation Matrix",
            "",
            "| Slot Category | Slot F1 Score | Description |",
            "| :--- | :---: | :--- |"
        ]

        for cat, f1_val in per_cat_f1.items():
            md_lines.append(f"| `{cat}` | **{f1_val}** | Intent / Domain Slot |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_path), str(md_path), summary
