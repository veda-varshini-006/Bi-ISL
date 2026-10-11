# Bi-ISL AR: Deployment & User Guide

**Project Title:** Bi-ISL AR: Real-Time Continuous Indian Sign Language to English Sentence Translation Using Live Camera and Augmented Reality Subtitles  
**Date:** October 2026  
**Repository:** [https://github.com/veda-varshini-006/Bi-ISL](https://github.com/veda-varshini-006/Bi-ISL)  

---

## 1. System Requirements & Prerequisites

### Server / Host Machine (Laptop or Workstation):
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware:**
  - Standard webcam (built-in or USB, $640 \times 480$ minimum resolution)
  - 8 GB RAM (16 GB recommended)
  - GPU (Optional): NVIDIA GPU with CUDA 11.8+ / 12.1+ for accelerated inference (< 100 ms)
- **Local Network:** Wi-Fi router or mobile hotspot for low-latency wireless communication between the host server and mobile AR device

### Mobile AR Device (Client):
- **Platform:** Android 8.0 (API Level 26) through Android 14 (API Level 34)
- **ARCore Support:** Google ARCore certified hardware (e.g., Google Pixel, Samsung Galaxy S series, OnePlus)
- **Camera:** Standard rear or front-facing RGB camera
- **Network:** Connected to the same local Wi-Fi network as the host laptop

---

## 2. Installation & Environment Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/veda-varshini-006/Bi-ISL.git
cd Bi-ISL

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Core Dependencies
Install the required machine learning, computer vision, and server dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # Or cu121 for GPU
pip install opencv-python mediapipe sentencepiece fastapi uvicorn websockets pydantic matplotlib pytest requests
```

### Step 3: Verify Checkpoints & Tokenizer
Verify that the model checkpoint and SentencePiece vocabulary exist in `artifacts/checkpoints/`:
- Checkpoint: `artifacts/checkpoints/PositionMotionTransformer_finetune_lr1e5_latest.pt` (115.3 MB)
- Tokenizer: `artifacts/checkpoints/BiISL_SentencePiece_5000.model` (327 KB)

To test environment readiness:
```bash
python scripts/verify_environment.py
```

---

## 3. Backend Server Setup & Streaming Service

The Bi-ISL AR backend runs as a high-concurrency FastAPI service handling binary JPEG video streaming, landmark extraction, and neural translation over WebSockets.

### Launching the Backend Server:
```bash
# Listen on all network interfaces (port 8000)
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

### Server Endpoints:
- **`GET /health`**: Health check returning server uptime, active checkpoint, and device status.
- **`GET /api/status`**: Detailed diagnostics including model parameter count and tokenizer state.
- **`POST /api/predict`**: Batch inference over raw landmark arrays.
- **`POST /api/translate_stream`**: REST streaming landmark ingestion returning provisional/finalized status.
- **`POST /api/benchmark`**: In-memory latency profiling over 50 iterations.
- **`POST /api/reset_stream`**: Resets circular buffers and clears active subtitles.
- **`WebSocket /ws/ar_stream`**: Bidirectional binary JPEG frame ingestion and real-time JSON subtitle emission.

### Network & Firewall Configuration:
To allow Android mobile devices to connect over Wi-Fi:
1. Find your host laptop's local IP address:
   - Windows: Run `ipconfig` (e.g., `192.168.1.15`)
   - Linux/macOS: Run `ifconfig` or `ip addr`
2. Allow inbound traffic on port 8000 through the Windows Defender / OS firewall:
   ```powershell
   New-NetFirewallRule -DisplayName "Bi-ISL AR Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```

---

## 4. Model Training & Offline Inference

### A. Model Training
To train or fine-tune the Pose-Motion Transformer on continuous ISL sentence data:

```bash
# 1. Quick verification training with synthetic fallback:
python scripts/train_pose_motion_transformer.py --epochs 5 --synthetic-fallback

# 2. Continuous ISL dataset training from manifest directory:
python scripts/train_pose_motion_transformer.py \
    --data-dir data/processed/cislr \
    --manifest dataset_manifest.json \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda  # or cpu
```

### B. Offline Video File Translation CLI
Translate continuous ISL video files and generate rendered MP4 videos with an embedded AR Subtitle HUD:
```bash
python scripts/translate_video.py --input sample_signing.mp4 --output subtitled_signing.mp4
```

### C. Live Desktop Camera Translation Preview
Test real-time camera translation directly on your host laptop without mobile hardware:
```bash
python scripts/translate_video.py --live
```
- Features a desktop HUD overlay displaying kinetic energy bars, FPS, confidence pills, and dynamic `[PROVISIONAL]` vs `[FINALIZED]` subtitle banners.

---

## 5. Unity Android AR Client Deployment

The mobile AR client application is located in the `unity/` directory.

### Requirements:
- **Unity Editor:** Version 2022.3 LTS or 2023.2 with **Android Build Support** (Android SDK & NDK tools installed).
- **AR Packages Configured (`unity/Packages/manifest.json`):**
  - `com.unity.xr.arfoundation`: 5.1.5
  - `com.unity.xr.arcore`: 5.1.5
  - `com.unity.textmeshpro`: 3.0.9
  - `com.unity.ugui`: 2.0.0

### Step-by-Step Build Instructions:
1. **Open Project in Unity:**
   - Launch Unity Hub.
   - Click **Open** and select the directory: `c:\Users\ADMIN\Downloads\Bi-ISL-main\Bi-ISL-main\unity`.
2. **Configure Android Build Settings:**
   - Go to **File $\to$ Build Settings...**
   - Under Platform, select **Android** and click **Switch Platform**.
   - Ensure the main AR scene (`Assets/Scenes/MainARScene.unity`) is added to "Scenes In Build".
3. **Verify Player Settings:**
   - Open **Edit $\to$ Project Settings $\to$ Player $\to$ Android**.
   - Under **Identification**:
     - Package Name: `com.biisl.ar`
     - Minimum API Level: **Android 8.0 'Oreo' (API Level 26)**
     - Target API Level: **Automatic (Highest Installed)** or **API Level 34**
   - Under **Configuration**:
     - Scripting Backend: **IL2CPP**
     - Target Architectures: **ARM64** checked
   - Under **XR Plug-in Management**:
     - Ensure **Google ARCore** is enabled under Android settings.
4. **Build APK Binary:**
   - Click **Build** (or **Build And Run** if your phone is connected via USB with Developer Mode and USB Debugging enabled).
   - Save the file as `BiISL_AR.apk`.

### In-App Mobile Usage:
1. Launch the app on your phone.
2. Grant the required **Camera** permission when prompted.
3. In the floating AR Control Panel, enter the host laptop's IP address (e.g., `192.168.1.15:8000`).
4. Tap **Connect**.
5. Frame the signer in the camera viewport. The AR HUD will project floating cyan `[PROVISIONAL]` drafts during active gestures and convert them to lime green `[FINALIZED]` sentences upon phrase pauses.

---

## 6. Android Native Jetpack Compose Client (`android/`)

For environments using native Android toolchains instead of Unity:
1. Open the `android/` directory in **Android Studio Hedgehog / Iguana**.
2. Sync the project with Gradle.
3. Ensure `usesCleartextTraffic="true"` is enabled in `AndroidManifest.xml` for local Wi-Fi development.
4. Deploy to device: Run **app** on connected device.

---

## 7. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Connection Refused (`WebSocket connection failed`)** | Server not running or blocked by firewall | Verify server is active with `curl http://<IP>:8000/health`. Add firewall inbound rule for port 8000. |
| **Black screen or no camera feed on device** | Missing camera runtime permission | Ensure `AndroidPermissionHelper.cs` prompts for `CAMERA` permission, or grant manually in Android App Settings. |
| **Subtitle flickers or clears immediately** | Insufficient hold duration | In `src/text/subtitle_stabilizer.py`, ensure `hold_duration_sec` is set to $4.0\text{s}$ (default). |
| **Subtitles overlap signer's gestures** | Signer-aware layout disabled | In Unity inspector on `ARSubtitleHUD`, ensure `SignerAwareLayout` script is attached and enabled. |
| **High CPU inference latency (> 1.0s)** | Running on low-power CPU | Enable CUDA GPU execution or reduce `max_new_tokens` in `translate()` from 39 to 20. |
