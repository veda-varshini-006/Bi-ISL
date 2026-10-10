"""Bi-ISL Streaming Translation Engine.

Manages temporal frame buffering, automatic sentence boundary detection,
sliding-window inference, provisional draft predictions, and finalized
sentence deduplication with desktop camera preview rendering.
"""

import time
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from src.models.pose_motion_transformer import BiISLTranslator
from src.vision.landmark_extractor import LandmarkExtractor
from src.vision.normalizer import LandmarkNormalizer
from src.vision.sentence_segmenter import SignActivityDetector
from src.text.subtitle_stabilizer import SubtitleStabilizer, ARSubtitle


@dataclass
class StreamingResult:
    """Output for a single streaming frame inference step."""
    active_text: str
    status: str  # "PROVISIONAL", "FINALIZED", or "IDLE"
    confidence: float
    is_signing: bool
    motion_energy: float
    buffered_frames: int
    finalized_text: Optional[str] = None
    provisional_text: Optional[str] = None
    is_new_finalized: bool = False


class StreamingTranslator:
    """Real-time streaming continuous ISL-to-English translation pipeline."""

    def __init__(
        self,
        translator: Optional[BiISLTranslator] = None,
        provisional_interval_frames: int = 12,
        min_inference_frames: int = 15,
        target_seq_len: int = 128,
        motion_threshold: float = 0.015,
        rest_pause_frames: int = 12
    ):
        self.translator = translator or BiISLTranslator.load_default()
        self.extractor = LandmarkExtractor()
        self.normalizer = LandmarkNormalizer()
        self.segmenter = SignActivityDetector(
            motion_threshold=motion_threshold,
            rest_pause_frames=rest_pause_frames,
            min_sign_frames=min_inference_frames,
            max_sign_frames=target_seq_len
        )
        self.stabilizer = SubtitleStabilizer(hold_duration_sec=4.5)

        self.provisional_interval_frames = provisional_interval_frames
        self.min_inference_frames = min_inference_frames
        self.target_seq_len = target_seq_len

        self.frame_counter = 0
        self.last_provisional_frame = 0
        self.last_motion_energy = 0.0

    def _interpolate_to_target(self, kps_list: List[np.ndarray]) -> np.ndarray:
        """Interpolates variable-length 75-keypoint sequence to target length (128)."""
        arr = np.array(kps_list, dtype=np.float32)  # (T, 75, 3)
        if len(arr) == self.target_seq_len:
            return arr

        tensor_raw = torch.from_numpy(arr).permute(1, 2, 0).unsqueeze(0)  # (1, 75, 3, T)
        tensor_flat = tensor_raw.reshape(1, 225, len(arr))
        tensor_resampled = torch.nn.functional.interpolate(
            tensor_flat, size=self.target_seq_len, mode="linear", align_corners=False
        )
        return tensor_resampled.squeeze(0).permute(1, 0).reshape(self.target_seq_len, 75, 3).numpy()

    def process_frame(
        self,
        frame_bgr_or_kps: Union[np.ndarray, List[Any]],
        timestamp_sec: Optional[float] = None
    ) -> StreamingResult:
        """Processes a single camera frame or pre-extracted landmark frame."""
        self.frame_counter += 1
        ts = timestamp_sec or time.time()

        # 1. Extract 75 keypoints
        if isinstance(frame_bgr_or_kps, np.ndarray) and frame_bgr_or_kps.ndim == 3 and frame_bgr_or_kps.shape[2] == 3 and frame_bgr_or_kps.shape[0] > 75:
            # Full BGR image
            raw_sample = self.extractor.extract_landmarks(frame_bgr_or_kps, frame_index=self.frame_counter, timestamp_sec=ts)
            norm_sample = self.normalizer.normalize_sample(raw_sample)

            p_kps = np.array(norm_sample.pose_landmarks, dtype=np.float32)[:, :3] if norm_sample.pose_landmarks else np.zeros((33, 3), dtype=np.float32)
            lh_kps = np.array(norm_sample.left_hand_landmarks, dtype=np.float32)[:, :3] if norm_sample.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
            rh_kps = np.array(norm_sample.right_hand_landmarks, dtype=np.float32)[:, :3] if norm_sample.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)

            if len(p_kps) < 33:
                p_kps = np.pad(p_kps, ((0, 33 - len(p_kps)), (0, 0)))
            if len(lh_kps) < 21:
                lh_kps = np.pad(lh_kps, ((0, 21 - len(lh_kps)), (0, 0)))
            if len(rh_kps) < 21:
                rh_kps = np.pad(rh_kps, ((0, 21 - len(rh_kps)), (0, 0)))

            kps_75 = np.vstack([p_kps[:33], lh_kps[:21], rh_kps[:21]])  # (75, 3)
        else:
            # Raw keypoint array
            raw_arr = np.array(frame_bgr_or_kps, dtype=np.float32)
            if raw_arr.ndim == 1:
                if len(raw_arr) >= 225:
                    kps_75 = raw_arr[:225].reshape(75, 3)
                else:
                    kps_75 = np.pad(raw_arr, (0, 225 - len(raw_arr))).reshape(75, 3)
            elif raw_arr.ndim == 2:
                if raw_arr.shape == (75, 3):
                    kps_75 = raw_arr
                elif raw_arr.shape[1] == 3 and len(raw_arr) >= 75:
                    kps_75 = raw_arr[:75]
                else:
                    kps_75 = raw_arr.flatten()[:225].reshape(75, 3)
            else:
                kps_75 = np.zeros((75, 3), dtype=np.float32)

        # 2. Track motion energy & sentence boundaries
        is_signing, completed_seq = self.segmenter.process_frame(kps_75)
        buffered_count = self.segmenter.current_buffer_length

        if self.segmenter.prev_landmarks is not None:
            self.last_motion_energy = self.segmenter.compute_frame_motion(kps_75, self.segmenter.prev_landmarks)

        provisional_text: Optional[str] = None
        finalized_text: Optional[str] = None
        is_new_finalized = False

        # 3. Check for Finalized Sentence Boundary
        if completed_seq is not None and len(completed_seq) >= self.min_inference_frames:
            interpolated_128 = self._interpolate_to_target(completed_seq)
            pred_text, conf = self.translator.translate(interpolated_128, return_confidence=True)

            if pred_text and len(pred_text.strip()) > 0:
                final_sub = self.stabilizer.update_finalized(pred_text, conf)
                if final_sub:
                    finalized_text = final_sub.text
                    is_new_finalized = True

        # 4. Check for Provisional Draft Update (while active signing)
        elif is_signing and buffered_count >= self.min_inference_frames:
            if (self.frame_counter - self.last_provisional_frame) >= self.provisional_interval_frames:
                self.last_provisional_frame = self.frame_counter
                current_buf = self.segmenter.active_buffer
                interpolated_128 = self._interpolate_to_target(current_buf)
                pred_text, conf = self.translator.translate(interpolated_128, return_confidence=True)

                if pred_text and len(pred_text.strip()) > 0:
                    prov_sub = self.stabilizer.update_provisional(pred_text, conf)
                    if prov_sub:
                        provisional_text = prov_sub.text

        # 5. Determine active presentation state
        active_sub = self.stabilizer.get_current()
        active_text = active_sub.text if active_sub else ""
        active_status = active_sub.status if active_sub else "IDLE"
        active_conf = active_sub.confidence if active_sub else 0.0

        return StreamingResult(
            active_text=active_text,
            status=active_status,
            confidence=round(float(active_conf), 3),
            is_signing=is_signing,
            motion_energy=round(float(self.last_motion_energy), 4),
            buffered_frames=buffered_count,
            finalized_text=finalized_text,
            provisional_text=provisional_text,
            is_new_finalized=is_new_finalized
        )

    def render_desktop_preview(
        self,
        frame: np.ndarray,
        result: StreamingResult,
        fps: float = 30.0
    ) -> np.ndarray:
        """Renders styled AR Subtitle HUD overlay onto video frame for desktop display."""
        if not CV2_AVAILABLE:
            return frame

        canvas = frame.copy()
        h, w, _ = canvas.shape
        font = cv2.FONT_HERSHEY_DUPLEX

        # 1. Floating AR HUD Box at bottom center
        box_h = 84
        box_y = h - box_h - 32
        overlay = canvas.copy()
        cv2.rectangle(overlay, (40, box_y), (w - 40, box_y + box_h), (11, 15, 25), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        # Border color depending on status
        if result.status == "FINALIZED":
            border_color = (16, 185, 129)  # Emerald green
            status_badge = "FINALIZED"
        elif result.status == "PROVISIONAL":
            border_color = (245, 158, 11)  # Amber / Cyan
            status_badge = "PROVISIONAL"
        else:
            border_color = (56, 189, 248)  # Sky blue
            status_badge = "LISTENING"

        cv2.rectangle(canvas, (40, box_y), (w - 40, box_y + box_h), border_color, 2)

        # 2. Status Badge & Metric Details
        cv2.putText(canvas, f"Bi-ISL AR | [{status_badge}]", (55, box_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)

        if result.confidence > 0:
            cv2.putText(canvas, f"Confidence: {int(result.confidence * 100)}%", (w - 240, box_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Subtitle Text
        display_text = result.active_text if result.active_text else "Waiting for signing..."
        text_scale = 0.82
        thickness = 2
        text_size = cv2.getTextSize(display_text, font, text_scale, thickness)[0]
        text_x = max(55, (w - text_size[0]) // 2)
        text_y = box_y + 54

        # Text color (dimmer if provisional, pure white if finalized)
        txt_color = (255, 255, 255) if result.status == "FINALIZED" else (220, 240, 255)
        cv2.putText(canvas, display_text, (text_x, text_y), font, text_scale, txt_color, thickness, cv2.LINE_AA)

        # 4. Kinetic Motion Energy Indicator Bar (Top Left)
        cv2.rectangle(canvas, (20, 20), (140, 36), (15, 23, 42), -1)
        energy_w = min(116, int(result.motion_energy * 2000))
        motion_col = (16, 185, 129) if result.is_signing else (100, 116, 139)
        cv2.rectangle(canvas, (22, 22), (22 + energy_w, 34), motion_col, -1)
        cv2.putText(canvas, "MOTION", (148, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    def reset(self):
        """Resets all streaming buffers and state."""
        self.segmenter.reset()
        self.stabilizer.clear()
        self.frame_counter = 0
        self.last_provisional_frame = 0
        self.last_motion_energy = 0.0
