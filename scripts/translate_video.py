#!/usr/bin/env python3
"""Bi-ISL Video Translation and Live AR Subtitles CLI.

Usage:
    python scripts/translate_video.py --input sample_signing.mp4 --output subtitled.mp4
    python scripts/translate_video.py --live
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.inference.video_translator import VideoTranslator
from src.vision.landmark_extractor import LandmarkExtractor
from src.vision.normalizer import LandmarkNormalizer
from src.vision.sentence_segmenter import SignActivityDetector
from src.text.subtitle_stabilizer import SubtitleStabilizer


def run_live_webcam_translation(translator: VideoTranslator, camera_id: int = 0):
    """Runs live webcam continuous sign language recognition with real-time AR Subtitle HUD."""
    print("=" * 65)
    print("  Bi-ISL AR: Real-Time Live Camera ISL-to-English Subtitles")
    print("  Press 'q' or ESC in the video window to stop.")
    print("=" * 65)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    extractor = LandmarkExtractor()
    normalizer = LandmarkNormalizer()
    segmenter = SignActivityDetector(
        motion_threshold=0.015,
        rest_pause_frames=12,
        min_sign_frames=15,
        max_sign_frames=128
    )
    stabilizer = SubtitleStabilizer(hold_time_sec=4.0)

    current_subtitle = "Position yourself in frame and start signing..."
    current_confidence = 0.0
    is_signing = False

    frame_idx = 0
    font = cv2.FONT_HERSHEY_DUPLEX

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            frame_idx += 1
            ts = time.time()

            # 1. Extract & normalize landmarks
            raw_sample = extractor.extract_landmarks(frame, frame_index=frame_idx, timestamp_sec=ts)
            norm_sample = normalizer.normalize_sample(raw_sample)

            p_kps = np.array(norm_sample.pose_landmarks, dtype=np.float32)[:, :3] if norm_sample.pose_landmarks else np.zeros((33, 3), dtype=np.float32)
            lh_kps = np.array(norm_sample.left_hand_landmarks, dtype=np.float32)[:, :3] if norm_sample.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
            rh_kps = np.array(norm_sample.right_hand_landmarks, dtype=np.float32)[:, :3] if norm_sample.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)

            if len(p_kps) < 33:
                p_kps = np.pad(p_kps, ((0, 33 - len(p_kps)), (0, 0)))
            if len(lh_kps) < 21:
                lh_kps = np.pad(lh_kps, ((0, 21 - len(lh_kps)), (0, 0)))
            if len(rh_kps) < 21:
                rh_kps = np.pad(rh_kps, ((0, 21 - len(rh_kps)), (0, 0)))

            combined_75 = np.vstack([p_kps[:33], lh_kps[:21], rh_kps[:21]])  # (75, 3)

            # 2. Segment signs & detect pauses
            sign_active, completed_seq = segmenter.process_frame(combined_75)
            is_signing = sign_active

            # 3. If a sign boundary / pause completed, run inference
            if completed_seq is not None and len(completed_seq) >= 15:
                # Resample to 128 frames
                arr = np.array(completed_seq, dtype=np.float32)
                tensor_raw = torch.from_numpy(arr).permute(1, 2, 0).unsqueeze(0)
                tensor_flat = tensor_raw.reshape(1, 225, len(arr))
                tensor_resampled = torch.nn.functional.interpolate(
                    tensor_flat, size=128, mode="linear", align_corners=False
                )
                arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()

                raw_trans, conf = translator.translator.translate(arr_128, return_confidence=True)
                if raw_trans and len(raw_trans.strip()) > 0:
                    stab = stabilizer.update(raw_trans, conf)
                    current_subtitle = stab.text if stab else raw_trans
                    current_confidence = conf
                    print(f"[{time.strftime('%H:%M:%S')}] ISL Translation: \"{current_subtitle}\" (Conf: {int(current_confidence*100)}%)")

            # 4. Render AR Subtitle HUD on screen
            box_h = 80
            box_y = h - box_h - 40
            overlay = frame.copy()
            cv2.rectangle(overlay, (50, box_y), (w - 50, box_y + box_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # AR HUD border
            border_color = (16, 185, 129) if is_signing else (56, 189, 248)
            cv2.rectangle(frame, (50, box_y), (w - 50, box_y + box_h), border_color, 2)

            # Subtitle text
            text_size = cv2.getTextSize(current_subtitle, font, 0.85, 2)[0]
            text_x = max(70, (w - text_size[0]) // 2)
            text_y = box_y + 48
            cv2.putText(frame, current_subtitle, (text_x, text_y), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # HUD Status Indicator
            status_text = "SIGNING DETECTED" if is_signing else "LISTENING / IDLE"
            cv2.putText(frame, f"AR Subtitles | Status: {status_text}", (65, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)

            if current_confidence > 0:
                cv2.putText(frame, f"Conf: {int(current_confidence * 100)}%", (w - 180, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Bi-ISL AR - Live Continuous ISL Translation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Bi-ISL Continuous Sign Language Translation CLI")
    parser.add_argument("--input", "-i", type=str, help="Path to input video (.mp4, .avi, .mov) or 'webcam'")
    parser.add_argument("--output", "-o", type=str, help="Path to save output video with AR subtitles overlay")
    parser.add_argument("--live", action="store_true", help="Launch live camera translation mode")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (.pt)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Custom SentencePiece model path (.model)")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cpu' or 'cuda')")

    args = parser.parse_args()

    print("=" * 65)
    print("           Bi-ISL AR: Continuous ISL Translation")
    print("=" * 65)

    translator = VideoTranslator(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device
    )

    if args.live or (args.input and args.input in ("webcam", "live", "camera", "0")):
        run_live_webcam_translation(translator, camera_id=args.camera_id)
        return

    if not args.input:
        print("[Error] Please specify --input <video_file> or --live")
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] File not found: {input_path}")
        sys.exit(1)

    print(f"[Info] Processing video: {input_path.name} ...")
    result = translator.translate_video(
        video_path=input_path,
        output_video_path=args.output
    )

    print("\n" + "-" * 65)
    print("TRANSLATION RESULTS:")
    print("-" * 65)
    print(f"  English Translation : {result['english_translation']}")
    print(f"  Confidence Score    : {result['confidence'] * 100:.1f}%")
    print(f"  Frames Analyzed     : {result['video_frames']} frames ({result['video_duration_sec']}s)")
    print(f"  Inference Latency   : {result['processing_latency_sec']}s (Real-Time Factor: {result['real_time_factor']}x)")
    if result['output_subtitled_video']:
        print(f"  Subtitled Video     : {result['output_subtitled_video']}")
    print("-" * 65)


if __name__ == "__main__":
    main()
