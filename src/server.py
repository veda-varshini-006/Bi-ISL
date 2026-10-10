"""Bi-ISL AR Real-Time Continuous ISL Translation FastAPI Backend Server.

Provides REST and real-time streaming endpoints for:
- Real-time continuous Indian Sign Language (ISL) to English sentence translation
- Pose-Motion Transformer with SentencePiece decoding (5000 vocab)
- Sign activity segmentation & AR subtitle stabilization
- Low-latency Wi-Fi communication for Android AR app and Web clients
- Hardware telemetry & inference benchmarking
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure src module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.pose_motion_transformer import BiISLTranslator
from src.text.subtitle_stabilizer import SubtitleStabilizer, ARSubtitle
from src.vision.sentence_segmenter import SignActivityDetector
from src.vision.landmark_extractor import LandmarkExtractor
from src.vision.normalizer import LandmarkNormalizer

app = FastAPI(
    title="Bi-ISL AR Real-Time Translation API",
    description="Continuous Indian Sign Language to English Sentence Translation Backend for Android AR & Web HUD",
    version="2.0.0"
)

# Enable CORS for Web UI and Mobile Clients on Local Wi-Fi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Translator, Vision Extractor & Stabilizer Instances
translator: Optional[BiISLTranslator] = None
stabilizer: SubtitleStabilizer = SubtitleStabilizer(hold_time_sec=4.0)
stream_segmenter: SignActivityDetector = SignActivityDetector(
    motion_threshold=0.015,
    rest_pause_frames=12,
    min_sign_frames=15,
    max_sign_frames=128
)
landmark_extractor: LandmarkExtractor = LandmarkExtractor()
normalizer: LandmarkNormalizer = LandmarkNormalizer()

model_load_status: str = "Uninitialized"
model_load_time_ms: float = 0.0
active_checkpoint_name: str = "PositionMotionTransformer_finetune_lr1e5_latest.pt"


def init_translation_engine():
    global translator, model_load_status, model_load_time_ms
    start_t = time.perf_counter()
    try:
        translator = BiISLTranslator.load_default(device=device)
        model_load_time_ms = round((time.perf_counter() - start_t) * 1000, 2)
        tok_status = "with SentencePiece (5k vocab)" if translator.sp is not None else "without tokenizer"
        model_load_status = f"Loaded Pose-Motion Transformer ({device.type.upper()}) {tok_status}"
        print(f"[{model_load_status}] loaded in {model_load_time_ms} ms.")
    except Exception as e:
        print(f"[Warning] Error initializing translation engine: {e}")
        model_load_status = f"Initialized Fallback Pose-Motion Engine ({device.type.upper()})"
        translator = BiISLTranslator(device=device)
        model_load_time_ms = round((time.perf_counter() - start_t) * 1000, 2)


@app.on_event("startup")
def startup_event():
    init_translation_engine()


# --- Pydantic Data Models ---

class LandmarkPredictRequest(BaseModel):
    landmarks: List[Any] = Field(
        ..., description="Continuous landmark sequence: [T, 75, 3] or [T, 225] or [T, 258] or 1D flat"
    )
    signer_id: Optional[str] = "signer_01"
    stabilize: Optional[bool] = True


class StreamingFrameRequest(BaseModel):
    landmarks: List[Any] = Field(
        ..., description="Single frame landmarks [75, 3] or [225] or [258]"
    )
    timestamp: Optional[float] = None
    signer_id: Optional[str] = "signer_01"


class TranslationResponse(BaseModel):
    english_sentence: str
    confidence: float
    is_stable: bool
    num_frames: int
    inference_latency_ms: float


# --- REST API Endpoints ---

@app.get("/health")
@app.get("/api/status")
def get_status():
    """Returns engine health, active model status, device, and tokenizer details."""
    vocab_size = translator.sp.get_piece_size() if (translator and translator.sp) else 5000
    has_sp = (translator is not None and translator.sp is not None)
    return {
        "status": "online",
        "service": "Bi-ISL AR Translation Engine",
        "model_status": model_load_status,
        "active_checkpoint": active_checkpoint_name,
        "device": device.type.upper(),
        "pytorch_version": torch.__version__,
        "tokenizer_active": has_sp,
        "vocab_size": vocab_size,
        "model_load_time_ms": model_load_time_ms
    }


@app.post("/api/predict")
def predict_isl(req: LandmarkPredictRequest):
    """Predicts continuous ISL English sentence translation from 3D landmark sequence."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    start_t = time.perf_counter()
    raw_landmarks = req.landmarks

    if not raw_landmarks:
        raise HTTPException(status_code=400, detail="Landmark sequence cannot be empty.")

    try:
        arr = np.array(raw_landmarks, dtype=np.float32)

        # Normalize shape
        if arr.ndim == 1:
            num_elements = len(arr)
            if num_elements % 225 == 0:
                frames = num_elements // 225
                arr = arr.reshape(frames, 75, 3)
            elif num_elements % 258 == 0:
                frames = num_elements // 258
                arr = arr.reshape(frames, 258)[:, :225].reshape(frames, 75, 3)
            else:
                frames = max(1, num_elements // 225)
                arr = np.pad(arr, (0, max(0, frames * 225 - num_elements)))[:frames * 225].reshape(frames, 75, 3)
        elif arr.ndim == 2:
            T, D = arr.shape
            if D == 225:
                arr = arr.reshape(T, 75, 3)
            elif D == 258:
                arr = arr[:, :225].reshape(T, 75, 3)
            elif D == 3 and T == 75:
                arr = arr.reshape(1, 75, 3)

        num_frames = arr.shape[0] if arr.ndim >= 2 else 1

        # Interpolate / resample to 128 frames
        if num_frames != 128:
            tensor_raw = torch.from_numpy(arr).to(dtype=torch.float32)
            if tensor_raw.dim() == 3:
                # (T, 75, 3) -> (1, 225, T)
                tensor_flat = tensor_raw.reshape(num_frames, 225).permute(1, 0).unsqueeze(0)
                tensor_resampled = torch.nn.functional.interpolate(
                    tensor_flat, size=128, mode="linear", align_corners=False
                )
                arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()
            else:
                arr_128 = np.zeros((128, 75, 3), dtype=np.float32)
        else:
            arr_128 = arr

        # Real autoregressive Transformer decoding
        raw_text, conf = translator.translate(arr_128, return_confidence=True)

        if req.stabilize:
            sub = stabilizer.update(raw_text, conf)
            final_text = sub.text if sub else raw_text
            is_stable = (sub is not None)
        else:
            final_text = raw_text
            is_stable = True

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "english_sentence": final_text,
            "raw_translation": raw_text,
            "confidence": round(float(conf), 4),
            "is_stable": is_stable,
            "num_frames_received": num_frames,
            "inference_latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/translate_stream")
def translate_stream(req: StreamingFrameRequest):
    """Processes incoming stream frames, segments sign activity, and updates AR subtitles."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    start_t = time.perf_counter()
    raw = np.array(req.landmarks, dtype=np.float32)

    # Format to (75, 3)
    if raw.ndim == 1:
        if len(raw) >= 225:
            frame_75 = raw[:225].reshape(75, 3)
        else:
            frame_75 = np.pad(raw, (0, 225 - len(raw))).reshape(75, 3)
    elif raw.ndim == 2:
        if raw.shape == (75, 3):
            frame_75 = raw
        elif raw.shape[1] == 3 and len(raw) >= 75:
            frame_75 = raw[:75]
        else:
            frame_75 = raw.flatten()[:225].reshape(75, 3)
    else:
        frame_75 = np.zeros((75, 3), dtype=np.float32)

    is_signing, completed_seq = stream_segmenter.process_frame(frame_75)

    translation_occurred = False
    provisional_text = None
    finalized_text = None
    status = "IDLE"

    # 1. Finalized sentence at boundary
    if completed_seq is not None and len(completed_seq) >= 15:
        arr = np.array(completed_seq, dtype=np.float32)
        tensor_raw = torch.from_numpy(arr).to(dtype=torch.float32)
        tensor_flat = tensor_raw.reshape(len(arr), 225).permute(1, 0).unsqueeze(0)
        tensor_resampled = torch.nn.functional.interpolate(
            tensor_flat, size=128, mode="linear", align_corners=False
        )
        arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()

        new_text, conf = translator.translate(arr_128, return_confidence=True)
        if new_text and len(new_text.strip()) > 0:
            final_sub = stabilizer.update_finalized(new_text, conf)
            if final_sub:
                finalized_text = final_sub.text
                status = "FINALIZED"
                translation_occurred = True

    # 2. Provisional draft while actively signing
    elif is_signing and stream_segmenter.current_buffer_length >= 15:
        status = "PROVISIONAL"
        if stream_segmenter.current_buffer_length % 12 == 0:
            current_buf = stream_segmenter.active_buffer
            arr = np.array(current_buf, dtype=np.float32)
            tensor_raw = torch.from_numpy(arr).to(dtype=torch.float32)
            tensor_flat = tensor_raw.reshape(len(arr), 225).permute(1, 0).unsqueeze(0)
            tensor_resampled = torch.nn.functional.interpolate(
                tensor_flat, size=128, mode="linear", align_corners=False
            )
            arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()
            prov_text, conf = translator.translate(arr_128, return_confidence=True)
            if prov_text and len(prov_text.strip()) > 0:
                prov_sub = stabilizer.update_provisional(prov_text, conf)
                if prov_sub:
                    provisional_text = prov_sub.text
                    translation_occurred = True

    current_sub = stabilizer.get_current_subtitle()
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return {
        "is_signing": is_signing,
        "status": current_sub.status if current_sub else status,
        "translation_updated": translation_occurred,
        "ar_subtitle": current_sub.text if current_sub else "",
        "ar_confidence": round(current_sub.confidence, 4) if current_sub else 0.0,
        "finalized_text": finalized_text,
        "provisional_text": provisional_text,
        "buffered_frames": stream_segmenter.current_buffer_length,
        "latency_ms": latency_ms
    }


landmark_extractor = LandmarkExtractor()
normalizer = LandmarkNormalizer()


@app.post("/api/reset_stream")
def reset_stream():
    """Resets stream buffer and active subtitles."""
    stream_segmenter.reset()
    stabilizer.clear()
    return {"status": "reset_successful"}


@app.websocket("/ws/ar_stream")
async def ar_stream_websocket(websocket: WebSocket):
    """High-performance WebSocket stream endpoint for Unity Android AR application.

    Receives camera frame JPEG bytes or landmark JSON packets over local Wi-Fi,
    streams back real-time provisional/finalized AR subtitles with signer bounding box.
    """
    await websocket.accept()
    print("[WebSocket] Unity AR client connected to /ws/ar_stream.")

    try:
        while True:
            message = await websocket.receive()
            start_t = time.perf_counter()

            kps_75 = None
            signer_bounds = {"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 1.0, "detected": False}

            # 1. Binary JPEG frame from camera
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if cv2 is not None:
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        raw_sample = landmark_extractor.extract_landmarks(frame, frame_index=0, timestamp_sec=time.time())
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

                        kps_75 = np.vstack([p_kps[:33], lh_kps[:21], rh_kps[:21]])

                        # Compute Signer Bounding Box in Normalized Viewport
                        all_pts = []
                        if norm_sample.pose_landmarks:
                            all_pts.extend([[float(p[0]), float(p[1])] for p in norm_sample.pose_landmarks if len(p) >= 2])
                        if norm_sample.left_hand_landmarks:
                            all_pts.extend([[float(p[0]), float(p[1])] for p in norm_sample.left_hand_landmarks if len(p) >= 2])
                        if norm_sample.right_hand_landmarks:
                            all_pts.extend([[float(p[0]), float(p[1])] for p in norm_sample.right_hand_landmarks if len(p) >= 2])

                        if all_pts:
                            pts_np = np.array(all_pts, dtype=np.float32)
                            signer_bounds = {
                                "min_x": float(np.clip(np.min(pts_np[:, 0]), 0.0, 1.0)),
                                "min_y": float(np.clip(np.min(pts_np[:, 1]), 0.0, 1.0)),
                                "max_x": float(np.clip(np.max(pts_np[:, 0]), 0.0, 1.0)),
                                "max_y": float(np.clip(np.max(pts_np[:, 1]), 0.0, 1.0)),
                                "detected": True
                            }

            # 2. JSON text message (pre-extracted landmarks or ping/command)
            elif "text" in message and message["text"]:
                import json
                try:
                    payload = json.loads(message["text"])
                    if payload.get("command") == "reset":
                        stream_segmenter.reset()
                        stabilizer.clear()
                        await websocket.send_json({"status": "reset_ack"})
                        continue
                    elif payload.get("command") == "ping":
                        await websocket.send_json({"status": "pong", "time": time.time()})
                        continue

                    raw_lms = payload.get("landmarks")
                    if raw_lms:
                        arr = np.array(raw_lms, dtype=np.float32)
                        if arr.ndim == 1 and len(arr) >= 225:
                            kps_75 = arr[:225].reshape(75, 3)
                        elif arr.ndim == 2 and arr.shape == (75, 3):
                            kps_75 = arr
                except Exception as e:
                    pass

            if kps_75 is None:
                kps_75 = np.zeros((75, 3), dtype=np.float32)

            # Process through sign activity segmenter
            is_signing, completed_seq = stream_segmenter.process_frame(kps_75)

            status = "IDLE"
            if completed_seq is not None and len(completed_seq) >= 15:
                arr = np.array(completed_seq, dtype=np.float32)
                tensor_raw = torch.from_numpy(arr).to(dtype=torch.float32)
                tensor_flat = tensor_raw.reshape(len(arr), 225).permute(1, 0).unsqueeze(0)
                tensor_resampled = torch.nn.functional.interpolate(
                    tensor_flat, size=128, mode="linear", align_corners=False
                )
                arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()

                new_text, conf = translator.translate(arr_128, return_confidence=True)
                if new_text and len(new_text.strip()) > 0:
                    final_sub = stabilizer.update_finalized(new_text, conf)
                    if final_sub:
                        status = "FINALIZED"
            elif is_signing and stream_segmenter.current_buffer_length >= 15:
                status = "PROVISIONAL"
                if stream_segmenter.current_buffer_length % 12 == 0:
                    current_buf = stream_segmenter.active_buffer
                    arr = np.array(current_buf, dtype=np.float32)
                    tensor_raw = torch.from_numpy(arr).to(dtype=torch.float32)
                    tensor_flat = tensor_raw.reshape(len(arr), 225).permute(1, 0).unsqueeze(0)
                    tensor_resampled = torch.nn.functional.interpolate(
                        tensor_flat, size=128, mode="linear", align_corners=False
                    )
                    arr_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(128, 75, 3).numpy()
                    prov_text, conf = translator.translate(arr_128, return_confidence=True)
                    if prov_text and len(prov_text.strip()) > 0:
                        prov_sub = stabilizer.update_provisional(prov_text, conf)

            current_sub = stabilizer.get_current_subtitle()
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            active_txt = current_sub.text if current_sub else ""
            active_st = current_sub.status if current_sub else status
            conf_val = round(float(current_sub.confidence), 3) if current_sub else 0.0

            await websocket.send_json({
                "active_text": active_txt,
                "text": active_txt,
                "status": active_st,
                "is_finalized": (active_st in ("FINALIZED", "CONFIRMED")),
                "confidence": conf_val,
                "is_signing": is_signing,
                "signer_bounds": signer_bounds,
                "buffered_frames": stream_segmenter.current_buffer_length,
                "latency_ms": latency_ms,
                "server_fps": 30.0,
                "activity_energy": 0.0
            })

    except WebSocketDisconnect:
        print("[WebSocket] Unity AR client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Stream error: {e}")


@app.post("/api/benchmark")
def run_benchmark():
    """Runs a 50-iteration inference benchmark to compute p50 and p95 latency."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    dummy_input = torch.randn(1, 128, 75, 3, device=device)
    dummy_tgt = torch.tensor([[2, 10, 20, 30]], dtype=torch.long, device=device)
    latencies = []

    # Warm-up (5 runs)
    for _ in range(5):
        with torch.no_grad():
            _ = translator.model(dummy_input, dummy_tgt)

    # Benchmark (50 runs)
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = translator.model(dummy_input, dummy_tgt)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = round(float(np.percentile(latencies, 50)), 2)
    p95 = round(float(np.percentile(latencies, 95)), 2)

    return {
        "benchmark_runs": 50,
        "device": device.type.upper(),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "min_latency_ms": round(latencies[0], 2),
        "max_latency_ms": round(latencies[-1], 2),
        "target_met": p95 < 200.0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
