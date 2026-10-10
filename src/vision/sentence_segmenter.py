"""Bi-ISL Automatic Sentence Segmentation and Sign Activity Detector.

Detects signing activity, hand stillness, and sentence boundaries in continuous
video/landmark streams to segment continuous ISL signing into discrete sentence buffers.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class SignActivityDetector:
    """Detects active signing vs. resting pauses from continuous landmark sequences."""

    def __init__(
        self,
        motion_threshold: float = 0.015,
        rest_duration_frames: int = 15,
        min_sentence_frames: int = 30,
        max_sentence_frames: int = 128,
        rest_pause_frames: Optional[int] = None,
        min_sign_frames: Optional[int] = None,
        max_sign_frames: Optional[int] = None
    ):
        self.motion_threshold = motion_threshold
        self.rest_duration_frames = rest_pause_frames if rest_pause_frames is not None else rest_duration_frames
        self.min_sentence_frames = min_sign_frames if min_sign_frames is not None else min_sentence_frames
        self.max_sentence_frames = max_sign_frames if max_sign_frames is not None else max_sentence_frames
        self.consecutive_rest_frames = 0
        self.active_buffer: List[np.ndarray] = []
        self.is_signing = False
        self.prev_landmarks: Optional[np.ndarray] = None

    @property
    def current_buffer_length(self) -> int:
        return len(self.active_buffer)

    def compute_frame_motion(self, current_kps: np.ndarray, prev_kps: np.ndarray) -> float:
        """Computes root mean square motion velocity across keypoints."""
        if current_kps is None or prev_kps is None or len(current_kps) == 0 or len(prev_kps) == 0:
            return 0.0
        diff = current_kps - prev_kps
        return float(np.sqrt(np.mean(diff ** 2)))

    def process_frame(
        self,
        landmarks: np.ndarray,
        prev_landmarks: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Processes a single frame.

        Returns:
            (is_signing, completed_sequence_or_none)
        """
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 3)

        effective_prev = prev_landmarks if prev_landmarks is not None else self.prev_landmarks

        motion = 0.0
        if effective_prev is not None:
            if effective_prev.ndim == 1:
                effective_prev = effective_prev.reshape(-1, 3)
            motion = self.compute_frame_motion(landmarks, effective_prev)

        self.prev_landmarks = landmarks.copy()

        # Determine if frame is actively signing
        is_active_frame = motion > self.motion_threshold

        if is_active_frame:
            self.is_signing = True
            self.consecutive_rest_frames = 0
            self.active_buffer.append(landmarks)
        else:
            if self.is_signing:
                self.consecutive_rest_frames += 1
                self.active_buffer.append(landmarks)

        # Check for sentence completion (resting boundary reached or max buffer limit)
        if len(self.active_buffer) >= self.min_sentence_frames:
            if self.consecutive_rest_frames >= self.rest_duration_frames or len(self.active_buffer) >= self.max_sentence_frames:
                # Completed sentence segment
                sentence_arr = np.array(self.active_buffer, dtype=np.float32)
                self.active_buffer = []
                self.is_signing = False
                self.consecutive_rest_frames = 0
                return False, sentence_arr

        return self.is_signing, None

    def flush(self) -> Optional[np.ndarray]:
        """Flushes any remaining buffered frames if above minimum length."""
        if len(self.active_buffer) >= self.min_sentence_frames:
            sentence_arr = np.array(self.active_buffer, dtype=np.float32)
            self.reset()
            return sentence_arr
        self.reset()
        return None

    def reset(self):
        """Clears all buffered frames and state."""
        self.active_buffer = []
        self.is_signing = False
        self.consecutive_rest_frames = 0
        self.prev_landmarks = None
