"""Integration tests for FastAPI server endpoints."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.server import app, init_translation_engine


@pytest.fixture(scope="module")
def client():
    init_translation_engine()
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_status_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "vocab_size" in data


def test_predict_endpoint(client):
    dummy_seq = np.random.randn(128, 75, 3).tolist()
    res = client.post("/api/predict", json={"landmarks": dummy_seq, "signer_id": "test_signer"})
    assert res.status_code == 200
    data = res.json()
    assert "english_sentence" in data
    assert "confidence" in data
    assert "inference_latency_ms" in data


def test_translate_stream_endpoint(client):
    dummy_frame = np.random.randn(75, 3).tolist()
    res = client.post("/api/translate_stream", json={"landmarks": dummy_frame})
    assert res.status_code == 200
    data = res.json()
    assert "is_signing" in data
    assert "ar_subtitle" in data


def test_benchmark_endpoint(client):
    res = client.post("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "p50_latency_ms" in data
    assert "p95_latency_ms" in data
    assert data["benchmark_runs"] == 50


def test_reset_stream_endpoint(client):
    res = client.post("/api/reset_stream")
    assert res.status_code == 200
    assert res.json()["status"] == "reset_successful"
