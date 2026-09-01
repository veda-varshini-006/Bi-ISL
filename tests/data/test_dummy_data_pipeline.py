"""Smoke tests for dummy data pipeline."""

import pytest
from src.data.schema import CanonicalDataSample
from src.data.adapter import BaseDatasetAdapter

class DummyDatasetAdapter(BaseDatasetAdapter):
    """Mock dataset adapter implementation for smoke testing."""
    def load_raw_dataset(self, dataset_path: str):
        return [
            {"id": "sample_1", "signer": "signer_01", "frames": 100, "text": "HELLO WORLD"},
            {"id": "sample_2", "signer": "signer_02", "frames": 150, "text": "THANK YOU"}
        ]

    def parse_samples(self, raw_data):
        samples = []
        for item in raw_data:
            samples.append(CanonicalDataSample(
                sample_id=item["id"],
                signer_id=item["signer"],
                frame_count=item["frames"],
                text_target=item["text"]
            ))
        return samples

def test_canonical_data_sample_schema():
    sample = CanonicalDataSample(
        sample_id="test_01",
        signer_id="signer_99",
        frame_count=120,
        text_target="GOOD MORNING"
    )
    assert sample.sample_id == "test_01"
    assert sample.signer_id == "signer_99"
    assert sample.frame_count == 120
    assert sample.text_target == "GOOD MORNING"

def test_dummy_data_pipeline_adapter():
    adapter = DummyDatasetAdapter()
    raw = adapter.load_raw_dataset("dummy/path")
    samples = adapter.parse_samples(raw)

    assert len(samples) == 2
    assert samples[0].sample_id == "sample_1"
    assert samples[1].signer_id == "signer_02"
