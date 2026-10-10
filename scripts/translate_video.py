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
from src.inference.streaming_translator import StreamingTranslator, StreamingResult


def run_live_webcam_translation(
    translator: VideoTranslator,
    camera_id: int = 0
):
    """Runs live webcam continuous sign language recognition with real-time desktop AR Subtitle HUD."""
    print("=" * 70)
    print("  Bi-ISL AR: Live Camera Continuous ISL-to-English Sentence Translation")
    print("  - Provisional Predictions : Active draft sentences while signing")
    print("  - Finalized Predictions   : Boundary-locked sentence translations")
    print("  - Deduplication           : Enabled across overlapping windows")
    print("  Press 'q' or ESC in the preview window to exit.")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    streamer = StreamingTranslator(
        translator=translator.translator,
        provisional_interval_frames=12,
        min_inference_frames=15,
        target_seq_len=128,
        motion_threshold=0.015,
        rest_pause_frames=12
    )

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - prev_time))
            prev_time = now

            # Process frame through streaming engine
            result = streamer.process_frame(frame, timestamp_sec=now)

            if result.is_new_finalized and result.finalized_text:
                print(f"[{time.strftime('%H:%M:%S')}] FINALIZED SENTENCE: \"{result.finalized_text}\" (Confidence: {int(result.confidence * 100)}%)")
            elif result.provisional_text:
                print(f"[{time.strftime('%H:%M:%S')}] Provisional Draft: \"{result.provisional_text}\" (Confidence: {int(result.confidence * 100)}%)", end="\r")

            # Render desktop preview with AR HUD
            preview_frame = streamer.render_desktop_preview(frame, result, fps=fps)

            # Show FPS counter top-right
            cv2.putText(preview_frame, f"FPS: {fps:.1f}", (preview_frame.shape[1] - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Bi-ISL AR - Live Desktop Continuous Translation", preview_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[Info] Live camera session ended.")


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

    print("=" * 70)
    print("           Bi-ISL AR: Continuous ISL Translation")
    print("=" * 70)

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

    print("\n" + "-" * 70)
    print("TRANSLATION RESULTS:")
    print("-" * 70)
    print(f"  English Translation : {result['english_translation']}")
    print(f"  Confidence Score    : {result['confidence'] * 100:.1f}%")
    print(f"  Frames Analyzed     : {result['video_frames']} frames ({result['video_duration_sec']}s)")
    print(f"  Inference Latency   : {result['processing_latency_sec']}s (Real-Time Factor: {result['real_time_factor']}x)")
    if result['output_subtitled_video']:
        print(f"  Subtitled Video     : {result['output_subtitled_video']}")
    print("-" * 70)


if __name__ == "__main__":
    main()
