# Bi-ISL AR: Unity Android AR Application

**Application Package:** `com.biisl.ar`  
**Target Platform:** Android (ARCore Compatible, Android 8.0+ / API 26-34)  
**XR Framework:** Unity AR Foundation 5.1.5 + Google ARCore XR Plugin  
**Rendering:** TextMeshPro UGUI over AR Camera Background  

---

## 1. System Overview

Bi-ISL AR turns an Android smartphone into a real-time assistive Augmented Reality visor for continuous Indian Sign Language (ISL) recognition:

```
[Phone Camera (AR Foundation)] 
       │ (Camera Frame JPEG @ 15 FPS)
       ▼ (Wi-Fi WebSocket / HTTP)
[Laptop FastAPI Backend (Port 8000)]
       │ (Pose-Motion Transformer + SentencePiece)
       ▼ (JSON Packet with Subtitles & Signer Bounds)
[Phone Screen: TextMeshPro AR HUD]
       │ (Signer-Aware Non-Occluding Adaptive Position)
       └─► Live English Sentence Subtitles!
```

---

## 2. Architecture & Components

| Component | File Path | Purpose |
| :--- | :--- | :--- |
| **AR Camera Manager** | `Assets/Scripts/AR/ARCameraBackgroundController.cs` | Acquires camera frames via `XRCpuImage`, downsamples to 480x640, and encodes to JPEG |
| **Network Client** | `Assets/Scripts/Networking/BiISLBackendClient.cs` | Bidirectional WebSocket stream (`ws://<ip>:8000/ws/ar_stream`) with automatic reconnection |
| **AR Subtitle HUD** | `Assets/Scripts/UI/ARSubtitleHUD.cs` | TextMeshPro renderer displaying provisional drafts, finalized sentences, confidence bar, and latency |
| **Signer-Aware Layout** | `Assets/Scripts/UI/SignerAwareLayout.cs` | Uses detected signer bounding box to dynamically shift subtitle HUD so hands/face are never occluded |
| **Control Panel** | `Assets/Scripts/UI/ARControlPanel.cs` | Start/Pause toggle, font size slider (20-42pt), caption visibility toggle, and Wi-Fi IP configuration |
| **Permissions** | `Assets/Scripts/Permissions/AndroidPermissionHelper.cs` | Android runtime camera & network permission request flow |

---

## 3. Unity Project Setup & Android Build Instructions

### Prerequisites
- **Unity Editor:** 2022.3 LTS (or 2023.2+) with **Android Build Support** (OpenJDK, Android SDK & NDK tools).
- **Physical Device:** Android device supporting Google Play Services for AR (ARCore).
- **Laptop Backend:** Python FastAPI backend running on local Wi-Fi network.

### Step 1: Open Project in Unity
1. Open **Unity Hub** $\to$ **Add Project from Disk** $\to$ select `unity/` folder.
2. Allow Unity to import packages (`com.unity.xr.arfoundation`, `com.unity.xr.arcore`, `com.unity.textmeshpro`).

### Step 2: Configure XR Plug-in Management
1. In Unity, go to **Edit** $\to$ **Project Settings** $\to$ **XR Plug-in Management**.
2. Under the **Android tab**, check **Google ARCore**.

### Step 3: Switch Platform to Android
1. Go to **File** $\to$ **Build Settings**.
2. Select **Android** $\to$ click **Switch Platform**.
3. Under **Target Architecture**, check **ARM64** (uncheck ARMv7).

### Step 4: Build APK
1. In **Build Settings**, click **Build** (or **Build and Run** if phone is connected via USB with USB Debugging enabled).
2. Save APK as `BiISL_AR.apk`.
3. If building manually via ADB:
   ```bash
   adb install -r BiISL_AR.apk
   ```

---

## 4. End-to-End Live Execution Guide

1. **Start Laptop Backend:**
   ```bash
   python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
   ```
   *(Note your laptop's local Wi-Fi IP address, e.g. `192.168.1.100` via `ipconfig`)*

2. **Launch Bi-ISL AR on Android:**
   - Open the app on your phone.
   - Accept the Camera permission prompt.
   - Tap the **Settings icon** $\to$ enter your laptop's IP (`192.168.1.100`) $\to$ tap **Apply / Connect**.
   - The status badge will change to **Connected to Bi-ISL Engine**.

3. **Continuous Sign Language Translation:**
   - Point the camera at a person signing in Indian Sign Language.
   - While actively signing: Amber `[PROVISIONAL]` subtitles stream in real time.
   - Upon completing the sign sentence / resting pause: Emerald `[FINALIZED]` subtitle locks in.
   - Subtitle HUD automatically shifts position if hands move into lower or upper viewports to prevent blocking sight of the sign kinematics.
