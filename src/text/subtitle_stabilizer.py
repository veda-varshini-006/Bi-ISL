"""Bi-ISL English Subtitle Stabilization and Smoothing Engine.

Prevents jitter, flickering, and duplicate text during streaming video translation inference.
Applies confidence gating, temporal persistence, Levenshtein similarity smoothing,
and dual-stage provisional vs finalized prediction management.
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
        status: str = "CONFIRMED"  # "PROVISIONAL" or "FINALIZED"
    ):
        self.text = text
        self.confidence = confidence
        self.timestamp = timestamp
        self.hold_duration_sec = hold_duration_sec
        self.status = status

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.hold_duration_sec

    @property
    def is_provisional(self) -> bool:
        return self.status == "PROVISIONAL"

    @property
    def is_finalized(self) -> bool:
        return self.status in ("FINALIZED", "CONFIRMED")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "status": self.status,
            "is_active": not self.is_expired,
            "is_finalized": self.is_finalized
        }


class SubtitleStabilizer:
    """Stabilizes real-time subtitles across overlapping translation inference windows.

    Features:
    - Provisional vs. Finalized subtitle separation
    - Overlapping window deduplication (suppresses repeated text from sliding frames)
    - Levenshtein similarity smoothing and confidence gating
    - Temporal hold persistence
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        hold_duration_sec: float = 4.0,
        similarity_threshold: float = 0.7,
        hold_time_sec: Optional[float] = None,
        dedup_window_sec: float = 6.0
    ):
        self.min_confidence = min_confidence
        self.hold_duration_sec = hold_time_sec if hold_time_sec is not None else hold_duration_sec
        self.similarity_threshold = similarity_threshold
        self.dedup_window_sec = dedup_window_sec
        self.active_subtitle: Optional[ARSubtitle] = None
        self.finalized_history: List[ARSubtitle] = []
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
                    dp[i - 1][j] + 1,        # deletion
                    dp[i][j - 1] + 1,        # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)

    def is_duplicate(self, candidate_text: str) -> bool:
        """Checks if the candidate text duplicates recently finalized subtitles."""
        if not candidate_text:
            return True

        now = time.time()
        c_clean = candidate_text.lower().strip()

        for item in reversed(self.finalized_history):
            if (now - item.timestamp) > self.dedup_window_sec:
                break
            prev_clean = item.text.lower().strip()
            # Direct match or substring or high similarity
            if c_clean == prev_clean:
                return True
            if self._levenshtein_similarity(c_clean, prev_clean) >= self.similarity_threshold:
                return True
            if len(c_clean) > 4 and len(prev_clean) > 4:
                if c_clean in prev_clean or prev_clean in c_clean:
                    return True

        return False

    def update_provisional(self, raw_text: str, confidence: float) -> Optional[ARSubtitle]:
        """Updates the live preview with an in-progress, provisional translation during signing."""
        raw_text = raw_text.strip()
        if not raw_text or confidence < self.min_confidence:
            return self.get_current()

        now = time.time()
        # Create or update provisional subtitle
        self.active_subtitle = ARSubtitle(
            text=raw_text,
            confidence=confidence,
            timestamp=now,
            hold_duration_sec=self.hold_duration_sec,
            status="PROVISIONAL"
        )
        return self.active_subtitle

    def update_finalized(self, raw_text: str, confidence: float) -> Optional[ARSubtitle]:
        """Locks in a completed sentence prediction at a sign pause or boundary.

        Deduplicates against overlapping sliding-window passes.
        """
        raw_text = raw_text.strip()
        if not raw_text or confidence < self.min_confidence:
            return self.get_current()

        # Check deduplication against recent finalized items
        if self.is_duplicate(raw_text):
            # Refresh timestamp of current active subtitle without creating a duplicate
            if self.active_subtitle:
                self.active_subtitle.timestamp = time.time()
                return self.active_subtitle
            return None

        now = time.time()
        final_sub = ARSubtitle(
            text=raw_text,
            confidence=confidence,
            timestamp=now,
            hold_duration_sec=self.hold_duration_sec,
            status="FINALIZED"
        )
        self.active_subtitle = final_sub
        self.finalized_history.append(final_sub)
        self.recent_history.append(raw_text)

        if len(self.finalized_history) > 50:
            self.finalized_history.pop(0)
        if len(self.recent_history) > 50:
            self.recent_history.pop(0)

        return self.active_subtitle

    def update(self, raw_text: str, confidence: float) -> Optional[ARSubtitle]:
        """Backward-compatible standard update (treats confident distinct text as finalized)."""
        raw_text = raw_text.strip()
        if not raw_text or confidence < self.min_confidence:
            if self.active_subtitle and self.active_subtitle.is_expired:
                self.active_subtitle = None
            return self.active_subtitle

        now = time.time()

        if self.active_subtitle is None:
            self.active_subtitle = ARSubtitle(
                text=raw_text,
                confidence=confidence,
                timestamp=now,
                hold_duration_sec=self.hold_duration_sec,
                status="FINALIZED"
            )
            self.finalized_history.append(self.active_subtitle)
            self.recent_history.append(raw_text)
            return self.active_subtitle

        similarity = self._levenshtein_similarity(raw_text, self.active_subtitle.text)

        if similarity >= self.similarity_threshold:
            # High similarity: update text if new prediction is longer / more complete
            if len(raw_text) >= len(self.active_subtitle.text) or confidence > self.active_subtitle.confidence:
                self.active_subtitle.text = raw_text
                self.active_subtitle.confidence = max(self.active_subtitle.confidence, confidence)
            self.active_subtitle.timestamp = now
        else:
            # Check if this new sentence is already in recent history
            if not self.is_duplicate(raw_text):
                self.active_subtitle = ARSubtitle(
                    text=raw_text,
                    confidence=confidence,
                    timestamp=now,
                    hold_duration_sec=self.hold_duration_sec,
                    status="FINALIZED"
                )
                self.finalized_history.append(self.active_subtitle)
                self.recent_history.append(raw_text)

        if len(self.recent_history) > 50:
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
        self.finalized_history = []
        self.recent_history = []
