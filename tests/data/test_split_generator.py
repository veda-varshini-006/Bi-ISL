"""Unit tests for Bi-ISL Deterministic Split Generation Subsystem."""

import os
import tempfile
import pytest

from src.data.schema import CanonicalDataSample
from src.data.split_generator import DeterministicSplitGenerator, SplitAlreadyExistsError


@pytest.fixture
def dummy_samples():
    """Create 20 dummy samples across 5 signers and 10 source videos."""
    samples = []
    for i in range(20):
        signer_id = f"signer_0{i % 5}"
        video_id = f"video_0{i % 10}"
        samples.append(CanonicalDataSample(
            sample_id=f"sample_{i:02d}",
            dataset="INCLUDE",
            source_video_id=video_id,
            signer_id=signer_id,
            text=f"text_{i}"
        ))
    return samples


def test_deterministic_seeded_splits(dummy_samples):
    """Test that split generation is bit-for-bit deterministic for a given seed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = DeterministicSplitGenerator(output_dir=tmp_dir)

        splits1, hashes1 = gen.generate_splits(dummy_samples, dataset_name="INCLUDE", seed=123, overwrite=True)

        # Clear file to allow second run in same temp dir
        manifest_file = os.path.join(tmp_dir, "include_signer_disjoint_s123_manifest.json")

        splits2, hashes2 = gen.generate_splits(dummy_samples, dataset_name="INCLUDE", seed=123, overwrite=True)

        assert hashes1 == hashes2
        assert [s.sample_id for s in splits1["train"]] == [s.sample_id for s in splits2["train"]]
        assert [s.sample_id for s in splits1["test"]] == [s.sample_id for s in splits2["test"]]


def test_signer_disjoint_splits(dummy_samples):
    """Test that protocol='signer_disjoint' guarantees zero signer overlap across splits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = DeterministicSplitGenerator(output_dir=tmp_dir)
        splits, hashes = gen.generate_splits(dummy_samples, dataset_name="INCLUDE", protocol="signer_disjoint", seed=42)

        train_signers = set(s.signer_id for s in splits["train"])
        val_signers = set(s.signer_id for s in splits["val"])
        test_signers = set(s.signer_id for s in splits["test"])

        assert len(train_signers.intersection(val_signers)) == 0
        assert len(train_signers.intersection(test_signers)) == 0
        assert len(val_signers.intersection(test_signers)) == 0


def test_source_video_disjoint_splits(dummy_samples):
    """Test that protocol='video_disjoint' guarantees zero source-video overlap across splits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = DeterministicSplitGenerator(output_dir=tmp_dir)
        splits, hashes = gen.generate_splits(dummy_samples, dataset_name="INCLUDE", protocol="video_disjoint", seed=42)

        train_vids = set(s.source_video_id for s in splits["train"])
        val_vids = set(s.source_video_id for s in splits["val"])
        test_vids = set(s.source_video_id for s in splits["test"])

        assert len(train_vids.intersection(val_vids)) == 0
        assert len(train_vids.intersection(test_vids)) == 0
        assert len(val_vids.intersection(test_vids)) == 0


def test_immutable_manifest_protection(dummy_samples):
    """Test that silent regeneration of existing manifests is blocked with SplitAlreadyExistsError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = DeterministicSplitGenerator(output_dir=tmp_dir)

        # Initial split generation
        gen.generate_splits(dummy_samples, dataset_name="INCLUDE", seed=99)

        # Second call without overwrite=True must raise SplitAlreadyExistsError
        with pytest.raises(SplitAlreadyExistsError, match="Silent regeneration is forbidden"):
            gen.generate_splits(dummy_samples, dataset_name="INCLUDE", seed=99, overwrite=False)
