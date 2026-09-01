"""Bi-ISL Deterministic Split Generation Subsystem (Prompt 16).

Implements deterministic train/validation/test split generation prioritizing:
1. Official split when published by dataset authors.
2. Source-video-disjoint split (no video clips cross split boundaries).
3. Signer-disjoint evaluation when signer metadata exists.
4. Deterministic seeded generation.

Writes immutable split manifests to `./data/splits/` and computes split hashes
for inclusion in experiment metadata. Never regenerates a split silently.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.data.schema import CanonicalDataSample
from src.data.split_auditor import SplitAuditor
from src.utils.logging import BiISLLogger


class SplitAlreadyExistsError(Exception):
    """Raised when attempting to silently overwrite an existing immutable split manifest."""
    pass


class DeterministicSplitGenerator:
    """Generator for creating reproducible, leak-free train/val/test split manifests."""

    def __init__(self, output_dir: str = "./data/splits", logger: Optional[BiISLLogger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or BiISLLogger(name="SplitGenerator")
        self.auditor = SplitAuditor(logger=self.logger)

    def generate_splits(
        self,
        samples: List[CanonicalDataSample],
        dataset_name: str,
        protocol: str = "signer_disjoint",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        overwrite: bool = False
    ) -> Tuple[Dict[str, List[CanonicalDataSample]], Dict[str, str]]:
        """Generate deterministic train/val/test splits and return (splits_dict, split_hashes)."""
        manifest_filename = f"{dataset_name.lower()}_{protocol}_s{seed}_manifest.json"
        manifest_path = self.output_dir / manifest_filename

        if manifest_path.exists() and not overwrite:
            msg = f"Immutable split manifest '{manifest_filename}' already exists at {manifest_path}. Silent regeneration is forbidden."
            self.logger.warning(msg)
            raise SplitAlreadyExistsError(msg)

        # 1. Check for official split priority
        has_official = all("split" in s.metadata for s in samples)
        if protocol == "official" and has_official:
            self.logger.info(f"Using official split annotations for {dataset_name}.")
            splits: Dict[str, List[CanonicalDataSample]] = {"train": [], "val": [], "test": []}
            for s in samples:
                split_name = str(s.metadata["split"]).lower()
                if split_name in ["val", "dev", "validation"]:
                    splits["val"].append(s)
                elif split_name == "test":
                    splits["test"].append(s)
                else:
                    splits["train"].append(s)

        # 2. Signer-disjoint priority
        elif protocol == "signer_disjoint" and any(s.signer_id for s in samples):
            self.logger.info(f"Generating signer-disjoint splits for {dataset_name} (seed={seed}).")
            signer_groups: Dict[str, List[CanonicalDataSample]] = {}
            for s in samples:
                signer_key = s.signer_id or "unknown_signer"
                signer_groups.setdefault(signer_key, []).append(s)

            signers = sorted(list(signer_groups.keys()))
            random.seed(seed)
            random.shuffle(signers)

            n_test = max(1, int(len(signers) * test_ratio))
            n_val = max(1, int(len(signers) * val_ratio))

            test_signers = set(signers[:n_test])
            val_signers = set(signers[n_test:n_test + n_val])
            train_signers = set(signers[n_test + n_val:])

            splits = {"train": [], "val": [], "test": []}
            for s in samples:
                signer_key = s.signer_id or "unknown_signer"
                if signer_key in test_signers:
                    splits["test"].append(s)
                elif signer_key in val_signers:
                    splits["val"].append(s)
                else:
                    splits["train"].append(s)

        # 3. Source-video-disjoint priority
        else:
            self.logger.info(f"Generating source-video-disjoint splits for {dataset_name} (seed={seed}).")
            video_groups: Dict[str, List[CanonicalDataSample]] = {}
            for s in samples:
                vid_key = s.source_video_id or s.sample_id
                video_groups.setdefault(vid_key, []).append(s)

            videos = sorted(list(video_groups.keys()))
            random.seed(seed)
            random.shuffle(videos)

            n_test = max(1, int(len(videos) * test_ratio))
            n_val = max(1, int(len(videos) * val_ratio))

            test_vids = set(videos[:n_test])
            val_vids = set(videos[n_test:n_test + n_val])
            train_vids = set(videos[n_test + n_val:])

            splits = {"train": [], "val": [], "test": []}
            for s in samples:
                vid_key = s.source_video_id or s.sample_id
                if vid_key in test_vids:
                    splits["test"].append(s)
                elif vid_key in val_vids:
                    splits["val"].append(s)
                else:
                    splits["train"].append(s)

        # Audit generated splits to ensure zero leakage
        self.auditor.audit_splits(splits, signer_disjoint_required=(protocol == "signer_disjoint"))

        # Save immutable split manifest and compute split hashes
        split_hashes = self._save_immutable_manifest(manifest_path, dataset_name, protocol, seed, splits)
        return splits, split_hashes

    def _save_immutable_manifest(
        self,
        manifest_path: Path,
        dataset_name: str,
        protocol: str,
        seed: int,
        splits: Dict[str, List[CanonicalDataSample]]
    ) -> Dict[str, str]:
        """Write immutable split manifest to disk and return split SHA-256 hashes."""
        manifest_data = {
            "dataset_name": dataset_name,
            "protocol": protocol,
            "seed": seed,
            "splits": {}
        }
        split_hashes = {}

        for split_name, sample_list in splits.items():
            sample_ids = [s.sample_id for s in sample_list]
            split_content_str = json.dumps(sorted(sample_ids))
            split_hash = hashlib.sha256(split_content_str.encode("utf-8")).hexdigest()
            split_hashes[split_name] = split_hash

            manifest_data["splits"][split_name] = {
                "count": len(sample_list),
                "sha256_hash": split_hash,
                "sample_ids": sample_ids
            }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        self.logger.info(f"Saved immutable split manifest to '{manifest_path}'", hashes=split_hashes)
        return split_hashes
