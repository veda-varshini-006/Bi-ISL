# Bi-ISL AR: Continuation Status & Architecture Report

**Project Title:** Bi-ISL AR: Real-Time Continuous Indian Sign Language to English Sentence Translation Using Live Camera and Augmented Reality Subtitles  
**Repository:** [https://github.com/veda-varshini-006/Bi-ISL](https://github.com/veda-varshini-006/Bi-ISL)  
**Date:** October 2026  
**Status:** In Active Continuation

---

## 1. Project Objective & Scope

Bi-ISL AR is dedicated to real-time, continuous Indian Sign Language (ISL) video and camera translation into natural English sentences, rendered as live Augmented Reality (AR) subtitles.

### Explicitly Excluded Components:
- **No 3D Signing Avatars:** All 3D meshes, blendshapes, Unity avatar viewers, and avatar rigging code have been permanently removed.
- **No English-to-ISL Conversion:** The pipeline is strictly unidirectional (ISL Camera/Video $\to$ English Sentences).
- **No Bidirectional Translation:** No text-to-pose synthesis or reverse neural pipelines.
- **No VR Headsets Required:** Built for Android mobile AR and standard laptop webcams.
- **No Hardcoded Predictions:** All decoded tokens originate dynamically from the autoregressive PyTorch Transformer and SentencePiece tokenizer.

---

## 2. Implemented & Verified Architecture

### A. Core Deep Learning & Tokenizer Pipeline
- **Translation Model (`src/models/pose_motion_transformer.py`):**
  - **PoseMotionEncoder:** 4-layer Transformer Encoder ($d_{model} = 256$, 8 heads, $d_{ff} = 1024$, norm_first) ingesting continuous 3D keypoints $(T, 75, 3)$ concatenated with temporal motion deltas $\Delta p_t$ $(T, 75 \times 6 = 450)$.
  - **Autoregressive Decoder:** 4-layer Transformer Decoder with causal self-attention generating English subword token embeddings.
  - **Tokenizer:** SentencePiece unigram/BPE model with a 5,000 token vocabulary (`artifacts/checkpoints/BiISL_SentencePiece_5000.model`).
  - **Translator Class (`BiISLTranslator`):** Handles sequence interpolation to $T=128$, autoregressive generation, softmax confidence scoring, and token decoding.

### B. Vision & Temporal Segmentation Pipeline
- **Landmark Extraction (`src/vision/landmark_extractor.py`):** Extracts 75 canonical landmarks (33 Pose + 21 Left Hand + 21 Right Hand) normalized per frame.
- **Coordinate Normalization (`src/vision/normalizer.py`):** Shoulder-midpoint translation and torso-length scale normalization.
- **Sign Activity Detector (`src/vision/sentence_segmenter.py`):** Detects continuous signing versus rest pauses using wrist/hand kinetic motion energy thresholds, outputting continuous sign segments for decoding.

### C. Subtitle Stabilization & Smoothing
- **Subtitle Stabilizer (`src/text/subtitle_stabilizer.py`):** Employs Levenshtein similarity filtering, confidence score thresholds, and temporal hold timers (default 4.0s) to eliminate flickering during real-time AR HUD presentation.

### D. End-to-End Inference & CLI
- **Video Translator (`src/inference/video_translator.py`):** End-to-end continuous video file translation and rendering of styled AR Subtitle HUD directly onto output MP4 videos.
- **CLI Translation Tool (`scripts/translate_video.py`):**
  - Continuous video file translation: `python scripts/translate_video.py --input video.mp4 --output subtitled.mp4`
  - Live webcam AR subtitles: `python scripts/translate_video.py --live`

### E. Backend FastAPI Server
- **Server (`src/server.py`):**
  - `GET /health` & `GET /api/status`: Real-time backend status, active checkpoint, device, and tokenizer details.
  - `POST /api/predict`: Real PyTorch model inference over 3D landmark arrays with subtitle stabilization.
  - `POST /api/translate_stream`: Streaming frame-by-frame landmark ingestion with sign activity detection.
  - `POST /api/benchmark`: 50-iteration inference latency profiling.
  - `POST /api/reset_stream`: Resets streaming buffers and active subtitles.

### F. Android AR Camera Application
- **Mobile Architecture (`android/`):**
  - `CameraScreen`: Jetpack Compose live viewfinder with glowing sign-activity indicator, floating AR Subtitle HUD, and confidence pill.
  - `TranscriptHistoryScreen`: Continuous transcript log of recognized sentences.
  - `SettingsScreen`: Wi-Fi server configuration to connect the phone to the laptop backend via local network (e.g. `http://192.168.1.5:8000`).
  - `RemoteInferenceClient.kt`: Low-latency asynchronous coroutine client.

---

## 3. Files Modified & Created

| Category | File Path | Status |
| :--- | :--- | :--- |
| **Model** | `src/models/pose_motion_transformer.py` | Implemented `PoseMotionTranslationModel`, `BiISLTranslator`, `load_default()`, confidence scoring |
| **Model** | `src/models/__init__.py` | Exported `PoseMotionTranslationModel` and `BiISLTranslator` |
| **Vision** | `src/vision/sentence_segmenter.py` | Implemented `SignActivityDetector` |
| **Text** | `src/text/subtitle_stabilizer.py` | Implemented `SubtitleStabilizer` & `ARSubtitle` |
| **Inference** | `src/inference/video_translator.py` | Implemented `VideoTranslator` with AR HUD overlay rendering |
| **CLI** | `scripts/translate_video.py` | Created CLI for video translation and live webcam AR subtitles |
| **Server** | `src/server.py` | Updated FastAPI server for continuous translation & streaming |
| **Android** | `android/app/src/main/java/com/biisl/app/inference/RemoteInferenceClient.kt` | Created Wi-Fi backend client |
| **Android** | `android/app/src/main/java/com/biisl/app/inference/InferenceEngine.kt` | Cleaned interface contracts |
| **Android** | `android/app/src/main/java/com/biisl/app/presentation/Screens.kt` | Built AR Subtitle HUD, Transcript, and Wi-Fi Settings |
| **Removed** | `src/avatar/`, `src/english_to_isl/`, `src/reverse/`, `unity/`, `android/.../avatar/` | Permanently purged |

---

## 4. Tests Executed & Verification Results

- **Unit & Integration Test Suite (`pytest`):**
  - Total tests executed: **218**
  - Total tests passed: **218 (100% passing in 31.8s)**
  - Newly added dedicated test suites:
    - `tests/models/test_pose_motion_transformer.py` (4 tests) - PoseMotionEncoder, PoseMotionTranslationModel, BiISLTranslator
    - `tests/text/test_subtitle_stabilizer.py` (4 tests) - SubtitleStabilizer, ARSubtitle, Levenshtein smoothing, hold duration
    - `tests/vision/test_sentence_segmenter.py` (3 tests) - SignActivityDetector, kinetic motion energy, pause segmentation
    - `tests/integration/test_server_endpoints.py` (5 tests) - FastAPI `/health`, `/api/status`, `/api/predict`, `/api/translate_stream`, `/api/benchmark`, `/api/reset_stream`
- **Model & Tokenizer Verification:**
  - `BiISLTranslator.load_default()` successfully verified.
  - SentencePiece 5000 vocabulary loaded and decoded tokens dynamically.

---

## 5. Commands to Run Working Components

### 1. Launch FastAPI Backend Server (on Laptop)
```bash
# Start backend server accessible to mobile devices over local Wi-Fi
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run CLI Video Translation
```bash
# Translate a video file and generate an AR subtitled output video
python scripts/translate_video.py --input sample_signing.mp4 --output sample_subtitled.mp4

# Run live webcam translation with on-screen AR Subtitles HUD
python scripts/translate_video.py --live
```

### 3. Run Web AR Subtitles HUD
```bash
# Serve the web application on port 8080
python -m http.server 8080 --directory web_app
# Open http://localhost:8080 in your browser
```

### 4. Build & Deploy Android AR App
```bash
cd android
./gradlew assembleDebug
# Deploy generated APK to Android device:
# adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## 6. Remaining Stages & Hardware Requirements

1. **Android On-Device Testing:** Test the live camera feed and Wi-Fi streaming latency with an Android phone connected to the local Wi-Fi network running the FastAPI backend.
2. **Domain Finetuning:** Collect additional continuous ISL sentences across diverse signers to further expand subword generation confidence.
