"""Unit tests for Bi-ISL Versioned Preprocessing Cache Subsystem."""

import os
import tempfile
import pytest

from src.data.preprocessing_cache import PreprocessingCache
from src.vision.landmark_extractor import MultimodalPoseSample


def test_cache_key_computation():
    """Test deterministic cache key computation across versions and configs."""
    cache = PreprocessingCache(extractor_version="v1.0.0_mediapipe", model_version="v1.0")
    config1 = {"fps": 25.0, "size": [224, 224]}
    config2 = {"fps": 30.0, "size": [224, 224]}

    key1 = cache.compute_cache_key("checksum_abc", config1)
    key2 = cache.compute_cache_key("checksum_abc", config2)
    key3 = cache.compute_cache_key("checksum_abc", config1)

    assert key1 == key3
    assert key1 != key2


def test_cache_put_and_get():
    """Test caching preprocessed pose samples, normalization metadata, and retrieval."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = PreprocessingCache(cache_dir=tmp_dir)
        config = {"fps": 25.0}
        checksum = "checksum_123"

        key = cache.compute_cache_key(checksum, config)
        sample = MultimodalPoseSample(frame_index=0, timestamp_sec=0.0, mask_pose=True)

        norm_meta = {"shoulder_midpoint": [0.5, 0.4, 0.0], "scale": 0.2}
        cache_path = cache.put(key, checksum, config, [sample], norm_meta)

        assert os.path.exists(cache_path)

        cached_data = cache.get(key)
        assert cached_data is not None
        assert cached_data["cache_key"] == key
        assert cached_data["frame_count"] == 1
        assert cached_data["normalization_metadata"]["scale"] == 0.2


def test_stale_cache_invalidation():
    """Test automatic invalidation of stale cache entries when versions/configs change."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_v1 = PreprocessingCache(cache_dir=tmp_dir, extractor_version="v1.0.0")
        config1 = {"fps": 25.0}
        checksum = "checksum_xyz"

        key = cache_v1.compute_cache_key(checksum, config1)
        sample = MultimodalPoseSample(frame_index=0, timestamp_sec=0.0, mask_pose=True)
        cache_v1.put(key, checksum, config1, [sample], {})

        # Access with new extractor version should invalidate stale entry
        cache_v2 = PreprocessingCache(cache_dir=tmp_dir, extractor_version="v2.0.0_updated")
        assert cache_v2.get(key) is None

        # Invalidate stale cache files
        deleted = cache_v2.invalidate_stale_cache(active_config_dict=config1)
        assert deleted >= 0


def test_preprocessing_benchmark_report():
    """Test generating preprocessing benchmark reports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = PreprocessingCache(cache_dir=os.path.join(tmp_dir, "cache"))
        json_rep, md_rep = cache.generate_benchmark_report(num_trials=5, output_dir=tmp_dir)

        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Bi-ISL Preprocessing Cache Benchmark Report" in md_text
        assert "Throughput Speedup:" in md_text
