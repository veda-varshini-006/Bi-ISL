"""Unit tests for Bi-ISL Canonical Data Schema and Dataset Translators."""

import pytest
from src.data.schema import (
    CanonicalDataSample,
    translate_include_sample,
    translate_isltranslate_sample,
    translate_isign_sample,
    translate_ishnews_sample
)


def test_canonical_data_sample_fields():
    """Test CanonicalDataSample schema fields and defaults."""
    sample = CanonicalDataSample(
        sample_id="test_001",
        dataset="INCLUDE",
        source_video_id="vid_101",
        segment_start=1.2,
        segment_end=5.4,
        signer_id="signer_05",
        video_path="/path/to/video.mp4",
        fps=30.0,
        width=1920,
        height=1080,
        text="thank you",
        gloss="THANK YOU",
        pose_path="/path/to/pose.npy",
        domain="isolated_sign",
        metadata={"custom_field": "val123"}
    )

    assert sample.sample_id == "test_001"
    assert sample.dataset == "INCLUDE"
    assert sample.source_video_id == "vid_101"
    assert sample.segment_start == 1.2
    assert sample.segment_end == 5.4
    assert sample.signer_id == "signer_05"
    assert sample.video_path == "/path/to/video.mp4"
    assert sample.fps == 30.0
    assert sample.width == 1920
    assert sample.height == 1080
    assert sample.text == "thank you"
    assert sample.gloss == "THANK YOU"
    assert sample.pose_path == "/path/to/pose.npy"
    assert sample.domain == "isolated_sign"
    assert sample.metadata["custom_field"] == "val123"


def test_include_translator():
    """Test translating raw INCLUDE sample into CanonicalDataSample."""
    raw = {
        "video_id": "v_042",
        "gloss": "WELCOME",
        "signer_id": "signer_12",
        "label_idx": 45,
        "category": "greetings"
    }

    sample = translate_include_sample(raw)
    assert sample.sample_id == "include_v_042"
    assert sample.dataset == "INCLUDE"
    assert sample.gloss == "WELCOME"
    assert sample.domain == "isolated_sign"
    assert sample.metadata["original_label_idx"] == 45
    assert sample.metadata["category"] == "greetings"


def test_isltranslate_translator():
    """Test translating raw ISLTranslate sample into CanonicalDataSample."""
    raw = {
        "id": "item_101",
        "english_translation": "Where is the nearest hospital?",
        "gloss": "HOSPITAL NEAREST WHERE",
        "topics": ["medical", "emergency"],
        "segment_start": 0.5,
        "segment_end": 4.8
    }

    sample = translate_isltranslate_sample(raw)
    assert sample.sample_id == "isltr_item_101"
    assert sample.dataset == "ISLTranslate"
    assert sample.text == "Where is the nearest hospital?"
    assert sample.gloss == "HOSPITAL NEAREST WHERE"
    assert sample.domain == "conversational"
    assert sample.metadata["topics"] == ["medical", "emergency"]


def test_isign_translator():
    """Test translating raw iSign sample into CanonicalDataSample."""
    raw = {
        "id": "isign_55",
        "text": "The weather is very pleasant today.",
        "gloss": "TODAY WEATHER PLEASANT",
        "pose_3d_available": True,
        "non_manual_features": {"eyebrow_raise": True}
    }

    sample = translate_isign_sample(raw)
    assert sample.sample_id == "isign_isign_55"
    assert sample.dataset == "iSign"
    assert sample.domain == "multi_task"
    assert sample.metadata["pose_3d_available"] is True
    assert sample.metadata["non_manual_features"]["eyebrow_raise"] is True


def test_ishnews_translator():
    """Test translating raw ISH-NEWS sample into CanonicalDataSample."""
    raw = {
        "id": "news_88",
        "english": "Government announced new digital initiatives.",
        "isl_gloss": "GOVT ANNOUNCE DIGITAL INITIATIVE NEW",
        "news_headline": "Digital India Expansion",
        "broadcast_date": "2026-03-15"
    }

    sample = translate_ishnews_sample(raw)
    assert sample.sample_id == "ishnews_news_88"
    assert sample.dataset == "ISH-NEWS"
    assert sample.domain == "news"
    assert sample.text == "Government announced new digital initiatives."
    assert sample.metadata["news_headline"] == "Digital India Expansion"
    assert sample.metadata["broadcast_date"] == "2026-03-15"
