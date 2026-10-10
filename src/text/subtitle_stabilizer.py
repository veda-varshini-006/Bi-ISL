"""Bi-ISL English Subtitle Stabilization and Smoothing Engine.

Prevents jitter, flickering, and duplicate text during streaming video translation inference.
Applies confidence gating, temporal persistence, and Levenshtein similarity smoothing.
"""

import time
from typing import Optional, Dict, Any, List


class ARSubtitle:
    """Represents an augmented reality subtitle item."""

    def __init__(
        self,
        text: str,
        confidence: float,
        timestamp: float,
        hold_duration_sec: float = 3.5,
        status: str = "CONFIRMED"
    ):
        self.text = text
        self.confidence = confidence
        self.timestamp = timestamp
        self.hold_duration_sec = hold_duration_sec
        self.status = status

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.hold_duration_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "status": self.status,
            "is_active": not self.is_expired
        }


class SubtitleStabilizer:
    """Stabilizes real-time subtitles across overlapping translation inference windows."""

    def __init__(
        self,
        min_confidence: float = 0.45,
        hold_duration_sec: float = 4.0,
        similarity_threshold: float = 0.7,
        hold_time_sec: Optional[float] = None
    ):
        self.min_confidence = min_confidence
        self.hold_duration_sec = hold_time_sec if hold_time_sec is not None else hold_duration_sec
        self.similarity_threshold = similarity_threshold
        self.active_subtitle: Optional[ARSubtitle] = None
        self.recent_history: List[str] = []

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Computes character-level similarity ratio between two strings [0.0 - 1.0]."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        s1_lower, s2_lower = s1.lower().strip(), s2.lower().strip()
        if s1_lower == s2_lower:
            return 1.0

        len1, len2 = len(s1_lower), len(s2_lower)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1_lower[i - 1] == s2_lower[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)

    def update(self, raw_text: str, confidence: float) -> Optional[ARSubtitle]:
        """Processes a new raw prediction from the translation model.

        Filters low-confidence results, suppresses minor phrasing fluctuations,
        and refreshes expiration timestamps.
        """
        raw_text = raw_text.strip()
        if not raw_text or confidence < self.min_confidence:
            # Check if active subtitle has expired
            if self.active_subtitle and self.active_subtitle.is_expired:
                self.active_subtitle = None
            return self.active_subtitle

        now = time.time()

        if self.active_subtitle is None:
            # New subtitle
            self.active_subtitle = ARSubtitle(
                text=raw_text,
                confidence=confidence,
                timestamp=now,
                hold_duration_sec=self.hold_duration_sec,
                status="CONFIRMED"
            )
            self.recent_history.append(raw_text)
            return self.active_subtitle

        # Check similarity with current active subtitle
        similarity = self._levenshtein_similarity(raw_text, self.active_subtitle.text)

        if similarity >= self.similarity_threshold:
            # High similarity: update text if new prediction is longer / more complete
            if len(raw_text) >= len(self.active_subtitle.text) or confidence > self.active_subtitle.confidence:
                self.active_subtitle.text = raw_text
                self.active_subtitle.confidence = max(self.active_subtitle.confidence, confidence)
            self.active_subtitle.timestamp = now  # Refresh hold time
        else:
            # New distinct sentence detected
            self.active_subtitle = ARSubtitle(
                text=raw_text,
                confidence=confidence,
                timestamp=now,
                hold_duration_sec=self.hold_duration_sec,
                status="CONFIRMED"
            )
            self.recent_history.append(raw_text)

        if len(self.recent_history) > 20:
            self.recent_history.pop(0)

        return self.active_subtitle

    def get_current(self) -> Optional[ARSubtitle]:
        """Returns active subtitle if still within hold duration."""
        if self.active_subtitle and not self.active_subtitle.is_expired:
            return self.active_subtitle
        return None

    def get_current_subtitle(self) -> Optional[ARSubtitle]:
        """Alias for get_current()."""
        return self.get_current()

    def clear(self):
        """Clears current active subtitle and history."""
        self.active_subtitle = None
        self.recent_history = []
