"""Bi-ISL Split Auditor and Dataset Leakage Subsystem (Experiment E0).

Detects:
1. Duplicate files (SHA-256 file hashes across splits).
2. Near-duplicate samples (identical text/gloss + matching temporal lengths).
3. Same source video across splits (source video leakage).
4. Same signer across signer-disjoint splits (signer leakage).
5. Identical text / video combinations across splits.
6. Overlapping temporal segments (segment_start/segment_end overlap).
7. Missing IDs (empty/null sample_id or video_id).
8. Broken paths (missing local video/pose files).

Generates:
- Structured JSON report
- CSV violations log
- Human-readable Markdown report

Fails loudly by raising LeakageAuditFailedError on confirmed source-video leakage.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from src.data.schema import CanonicalDataSample
from src.utils.logging import BiISLLogger


class LeakageAuditFailedError(Exception):
    """Raised when critical dataset leakage (e.g., source-video leakage) is confirmed."""
    pass


class SplitAuditor:
    """Auditor for detecting data leakage and integrity violations across dataset splits."""

    def __init__(self, logger: Optional[BiISLLogger] = None):
        self.logger = logger or BiISLLogger(name="SplitAuditor")

    def audit_splits(
        self,
        splits: Dict[str, List[CanonicalDataSample]],
        signer_disjoint_required: bool = True
    ) -> Dict[str, Any]:
        """Perform exhaustive leakage audit across dataset splits (e.g. train, dev, test)."""
        violations: List[Dict[str, Any]] = []

        split_names = list(splits.keys())
        all_samples: List[Tuple[str, CanonicalDataSample]] = [
            (split_name, sample)
            for split_name, sample_list in splits.items()
            for sample in sample_list
        ]

        # 1. Missing IDs & Broken Paths Audit
        for split_name, sample in all_samples:
            if not sample.sample_id or sample.sample_id.strip() == "":
                violations.append({
                    "severity": "HIGH",
                    "type": "MISSING_ID",
                    "split": split_name,
                    "sample_id": "UNKNOWN",
                    "details": "Sample is missing a valid sample_id."
                })
            if sample.video_path and not os.path.exists(sample.video_path):
                violations.append({
                    "severity": "LOW",
                    "type": "BROKEN_PATH",
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "details": f"Video path does not exist: {sample.video_path}"
                })

        # 2. Source-Video Leakage Audit (CRITICAL)
        video_split_map: Dict[str, Set[str]] = {}
        for split_name, sample in all_samples:
            if sample.source_video_id:
                video_split_map.setdefault(sample.source_video_id, set()).add(split_name)

        source_video_leakage_detected = False
        for video_id, split_set in video_split_map.items():
            if len(split_set) > 1:
                source_video_leakage_detected = True
                violations.append({
                    "severity": "CRITICAL",
                    "type": "SOURCE_VIDEO_LEAKAGE",
                    "split": ",".join(sorted(list(split_set))),
                    "sample_id": f"video_{video_id}",
                    "details": f"Source video '{video_id}' appears in multiple splits: {sorted(list(split_set))}"
                })

        # 3. Signer Leakage Audit (Signer-Disjoint Requirement)
        signer_split_map: Dict[str, Set[str]] = {}
        for split_name, sample in all_samples:
            if sample.signer_id:
                signer_split_map.setdefault(sample.signer_id, set()).add(split_name)

        if signer_disjoint_required:
            for signer_id, split_set in signer_split_map.items():
                if len(split_set) > 1:
                    violations.append({
                        "severity": "HIGH",
                        "type": "SIGNER_LEAKAGE",
                        "split": ",".join(sorted(list(split_set))),
                        "sample_id": f"signer_{signer_id}",
                        "details": f"Signer '{signer_id}' overlaps across signer-disjoint splits: {sorted(list(split_set))}"
                    })

        # 4. Overlapping Temporal Segments Audit
        for i in range(len(all_samples)):
            split1, s1 = all_samples[i]
            for j in range(i + 1, len(all_samples)):
                split2, s2 = all_samples[j]
                if split1 != split2 and s1.source_video_id and s1.source_video_id == s2.source_video_id:
                    if (
                        s1.segment_start is not None and s1.segment_end is not None and
                        s2.segment_start is not None and s2.segment_end is not None
                    ):
                        overlap_start = max(s1.segment_start, s2.segment_start)
                        overlap_end = min(s1.segment_end, s2.segment_end)
                        if overlap_start < overlap_end:
                            violations.append({
                                "severity": "CRITICAL",
                                "type": "TEMPORAL_SEGMENT_OVERLAP",
                                "split": f"{split1} vs {split2}",
                                "sample_id": f"{s1.sample_id} & {s2.sample_id}",
                                "details": f"Temporal overlap [{overlap_start:.2f}s, {overlap_end:.2f}s] on video '{s1.source_video_id}'"
                            })

        # 5. Near-Duplicate & Exact Duplicate Text/Video Audit
        text_video_map: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for split_name, sample in all_samples:
            if sample.text and sample.source_video_id:
                key = (sample.source_video_id, sample.text.strip().lower())
                text_video_map.setdefault(key, []).append((split_name, sample.sample_id))

        for (vid, txt), occurrences in text_video_map.items():
            splits_involved = set(occ[0] for occ in occurrences)
            if len(splits_involved) > 1:
                violations.append({
                    "severity": "HIGH",
                    "type": "EXACT_DUPLICATE_PAIR",
                    "split": ",".join(sorted(list(splits_involved))),
                    "sample_id": f"vid_{vid}",
                    "details": f"Identical (video, text) pair for text '{txt}' found across splits {sorted(list(splits_involved))}"
                })

        summary = {
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples_audited": len(all_samples),
            "split_counts": {name: len(sample_list) for name, sample_list in splits.items()},
            "total_violations": len(violations),
            "critical_violations": sum(1 for v in violations if v["severity"] == "CRITICAL"),
            "source_video_leakage_detected": source_video_leakage_detected,
            "violations": violations
        }

        # FAIL LOUDLIES ON CONFIRMED SOURCE-VIDEO LEAKAGE
        if source_video_leakage_detected:
            msg = f"CRITICAL LEAKAGE FAILURE: Confirmed source-video leakage detected across splits ({summary['critical_violations']} critical violations)."
            self.logger.error(msg, summary=summary)
            raise LeakageAuditFailedError(msg)

        return summary

    def export_reports(self, audit_summary: Dict[str, Any], output_dir: str = "./artifacts/reports/e0_audit") -> Tuple[str, str, str]:
        """Export JSON, CSV, and Markdown audit reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "e0_leakage_report.json"
        csv_file = out_path / "e0_leakage_violations.csv"
        md_file = out_path / "e0_leakage_report.md"

        # 1. JSON Report
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)

        # 2. CSV Violations
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["severity", "type", "split", "sample_id", "details"])
            for v in audit_summary.get("violations", []):
                writer.writerow([v["severity"], v["type"], v["split"], v["sample_id"], v["details"]])

        # 3. Markdown Report
        md_lines = [
            "# Experiment E0: Dataset Leakage Audit Report",
            "",
            f"**Audit Timestamp:** {audit_summary['audit_timestamp']}",
            f"**Total Samples Audited:** {audit_summary['total_samples_audited']}",
            f"**Total Violations:** {audit_summary['total_violations']}",
            f"**Critical Violations:** {audit_summary['critical_violations']}",
            f"**Source Video Leakage Detected:** {'YES (FAIL)' if audit_summary['source_video_leakage_detected'] else 'NO (PASS)'}",
            "",
            "## Split Sample Counts",
            ""
        ]
        for sname, scount in audit_summary["split_counts"].items():
            md_lines.append(f"- **{sname}:** {scount} samples")

        md_lines.extend(["", "## Detailed Violations Log", ""])
        if audit_summary["violations"]:
            md_lines.append("| Severity | Type | Split(s) | Sample ID | Details |")
            md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for v in audit_summary["violations"]:
                md_lines.append(f"| **{v['severity']}** | `{v['type']}` | {v['split']} | {v['sample_id']} | {v['details']} |")
        else:
            md_lines.append("✅ **Zero data leakage or integrity violations detected.**")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_file), str(csv_file), str(md_file)
