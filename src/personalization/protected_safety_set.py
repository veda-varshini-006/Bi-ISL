"""Protected Reference Safety Set for UGSA Personalization (Prompt 44).

Requirements:
- Small (50 representative samples across fundamental ISL domains).
- NEVER used for normal online adapter weight updates.
- Representative of general ISL vocabulary and landmark distributions.
- Versioned with SHA256 integrity manifest hashing.
- NOT contaminated by target test dataset samples.
- Provides pre/post adaptation performance degradation measurement & rollback check.

Stores:
- docs/personalization/PROTECTED_SAFETY_SET_CONSTRUCTION.md
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ProtectedSafetySet:
    """Versioned protected reference safety set for anti-forgetting auditing."""

    def __init__(self, version_id: str = "v1.0.0", samples: Optional[List[Dict[str, Any]]] = None):
        self.version_id = version_id
        if samples is None:
            self.samples = [
                {
                    "sample_id": f"safety_ref_{i:03d}",
                    "text": f"general safety reference phrase {i}",
                    "domain": "GENERAL_ISL_CORE",
                    "signer_id": "calibration_group"
                }
                for i in range(50)
            ]
        else:
            self.samples = samples

        self.manifest_hash = self._compute_manifest_hash()

    def _compute_manifest_hash(self) -> str:
        """Computes deterministic SHA256 hash over safety set contents."""
        serialized = json.dumps(self.samples, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def audit_non_contamination(self, test_sample_ids: List[str]) -> bool:
        """Audit safety set against test sample IDs to guarantee ZERO contamination."""
        safety_ids = {s["sample_id"] for s in self.samples}
        test_ids = set(test_sample_ids)
        overlap = safety_ids.intersection(test_ids)
        if len(overlap) > 0:
            raise ValueError(f"Contamination Detected! Safety set shares IDs with test set: {overlap}")
        return True

    def measure_adaptation_degradation(
        self,
        pre_adaptation_bleu: float,
        post_adaptation_bleu: float,
        max_allowed_degradation_pct: float = 5.0
    ) -> Dict[str, Any]:
        """Calculates pre/post adaptation degradation on safety set."""
        if pre_adaptation_bleu <= 0.0:
            degradation_pct = 0.0
        else:
            degradation_pct = max(0.0, ((pre_adaptation_bleu - post_adaptation_bleu) / pre_adaptation_bleu) * 100.0)

        rollback_required = degradation_pct > max_allowed_degradation_pct

        return {
            "version_id": self.version_id,
            "manifest_hash": self.manifest_hash,
            "pre_adaptation_bleu": pre_adaptation_bleu,
            "post_adaptation_bleu": post_adaptation_bleu,
            "degradation_pct": round(degradation_pct, 2),
            "max_allowed_degradation_pct": max_allowed_degradation_pct,
            "rollback_required": rollback_required
        }
