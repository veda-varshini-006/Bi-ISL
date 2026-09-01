"""Bi-ISL Unified Canonical Data Schema and Dataset Translators.

Defines a unified internal sample representation (CanonicalDataSample) for all ISL benchmark datasets:
- INCLUDE
- ISLTranslate
- iSign
- ISH-NEWS

Ensures that dataset-specific fields remain recoverable inside the metadata dictionary.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class CanonicalDataSample(BaseModel):
    """Unified internal sample schema for Bi-ISL multimodal data pipelines."""

    sample_id: str
    dataset: str
    source_video_id: Optional[str] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    signer_id: Optional[str] = None
    video_path: Optional[str] = None
    fps: Optional[float] = 30.0
    width: Optional[int] = 1280
    height: Optional[int] = 720
    text: Optional[str] = None
    gloss: Optional[str] = None
    pose_path: Optional[str] = None
    domain: Optional[str] = "general"
    frame_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def translate_include_sample(raw_item: Dict[str, Any]) -> CanonicalDataSample:
    """Translate raw INCLUDE isolated sign sample into CanonicalDataSample."""
    sample_id = str(raw_item.get("video_id", raw_item.get("sample_id", "include_001")))
    gloss = raw_item.get("gloss", raw_item.get("label", ""))
    signer_id = raw_item.get("signer_id", "unknown_signer")
    video_path = raw_item.get("video_path", f"./data/raw/INCLUDE/videos/{sample_id}.mp4")

    extra_metadata = {
        "original_label_idx": raw_item.get("label_idx"),
        "category": raw_item.get("category", "isolated"),
        "split": raw_item.get("split", "train")
    }

    return CanonicalDataSample(
        sample_id=f"include_{sample_id}",
        dataset="INCLUDE",
        source_video_id=sample_id,
        signer_id=signer_id,
        video_path=video_path,
        fps=raw_item.get("fps", 30.0),
        width=raw_item.get("width", 1920),
        height=raw_item.get("height", 1080),
        text=raw_item.get("text", gloss.lower()),
        gloss=gloss,
        pose_path=raw_item.get("pose_path", f"./data/processed/keypoints/include_{sample_id}.npy"),
        domain="isolated_sign",
        metadata=extra_metadata
    )


def translate_isltranslate_sample(raw_item: Dict[str, Any]) -> CanonicalDataSample:
    """Translate raw ISLTranslate continuous translation sample into CanonicalDataSample."""
    sample_id = str(raw_item.get("id", raw_item.get("sample_id", "isltr_001")))
    text = raw_item.get("text", raw_item.get("english_translation", ""))
    gloss = raw_item.get("gloss", "")

    extra_metadata = {
        "topics": raw_item.get("topics", []),
        "grammar_tags": raw_item.get("grammar_tags", []),
        "split": raw_item.get("split", "train")
    }

    return CanonicalDataSample(
        sample_id=f"isltr_{sample_id}",
        dataset="ISLTranslate",
        source_video_id=raw_item.get("video_id", sample_id),
        segment_start=raw_item.get("segment_start", 0.0),
        segment_end=raw_item.get("segment_end"),
        signer_id=raw_item.get("signer_id", "signer_cfilt"),
        video_path=raw_item.get("video_path"),
        fps=raw_item.get("fps", 25.0),
        width=raw_item.get("width", 1280),
        height=raw_item.get("height", 720),
        text=text,
        gloss=gloss,
        pose_path=raw_item.get("pose_path"),
        domain="conversational",
        metadata=extra_metadata
    )


def translate_isign_sample(raw_item: Dict[str, Any]) -> CanonicalDataSample:
    """Translate raw iSign multi-task sample into CanonicalDataSample."""
    sample_id = str(raw_item.get("id", "isign_001"))
    text = raw_item.get("text", "")
    gloss = raw_item.get("gloss", "")

    extra_metadata = {
        "pose_3d_available": raw_item.get("pose_3d_available", True),
        "non_manual_features": raw_item.get("non_manual_features", {}),
        "split": raw_item.get("split", "train")
    }

    return CanonicalDataSample(
        sample_id=f"isign_{sample_id}",
        dataset="iSign",
        source_video_id=raw_item.get("video_id", sample_id),
        signer_id=raw_item.get("signer_id"),
        video_path=raw_item.get("video_path"),
        fps=raw_item.get("fps", 30.0),
        width=raw_item.get("width", 1280),
        height=raw_item.get("height", 720),
        text=text,
        gloss=gloss,
        pose_path=raw_item.get("pose_path"),
        domain="multi_task",
        metadata=extra_metadata
    )


def translate_ishnews_sample(raw_item: Dict[str, Any]) -> CanonicalDataSample:
    """Translate raw ISH-NEWS continuous news-domain sample into CanonicalDataSample."""
    sample_id = str(raw_item.get("id", "ishnews_001"))
    text = raw_item.get("english", raw_item.get("text", ""))
    gloss = raw_item.get("isl_gloss", raw_item.get("gloss", ""))

    extra_metadata = {
        "news_headline": raw_item.get("news_headline", ""),
        "broadcast_date": raw_item.get("broadcast_date", "2026-01-01"),
        "split": raw_item.get("split", "train")
    }

    return CanonicalDataSample(
        sample_id=f"ishnews_{sample_id}",
        dataset="ISH-NEWS",
        source_video_id=raw_item.get("video_id", sample_id),
        signer_id=raw_item.get("signer_id", "news_anchor"),
        video_path=raw_item.get("video_path"),
        fps=raw_item.get("fps", 25.0),
        width=raw_item.get("width", 1920),
        height=raw_item.get("height", 1080),
        text=text,
        gloss=gloss,
        pose_path=raw_item.get("pose_path"),
        domain="news",
        metadata=extra_metadata
    )
