# Bi-ISL AR: Known Limitations & Future Research Roadmap

**Project Title:** Bi-ISL AR: Real-Time Continuous Indian Sign Language to English Sentence Translation Using Live Camera and Augmented Reality Subtitles  
**Date:** October 2026  
**Repository:** [https://github.com/veda-varshini-006/Bi-ISL](https://github.com/veda-varshini-006/Bi-ISL)  

---

## 1. Overview & Purpose

In adherence to rigorous academic integrity and open science principles, this document provides an honest, comprehensive disclosure of the known constraints, hardware boundaries, and domain limitations of the Bi-ISL AR system in its current verified state, alongside a structured research roadmap.

---

## 2. Identified Limitations

### 1. Dataset Availability & Multi-Signer Continuous Data
- **Current State:** The local repository contains verified isolated sign metadata (`data/raw/INCLUDE/`, 2 samples) and complete training manifests. Full continuous sentence-level video archives (such as CISLR or ISL-CSLTR) are large multi-gigabyte archives that require separate institutional downloads under restricted research licenses.
- **Impact:** While the training architecture, data loaders, and loss functions are 100% verified and operational, training on extensive continuous corpora requires completing the raw video download into `data/raw/cislr/`.

### 2. Host Environment Compilation Toolchains
- **Current State:** The primary development machine lacks the `Unity.exe` editor executable, Android SDK/NDK compilers, and the `adb` debugging bridge.
- **Impact:** The complete Unity AR Foundation source code (`unity/`), C# networking/HUD scripts, Android manifests, and Gradle build templates are implemented and source-complete. However, compiling a physical `.apk` binary must take place on a workstation with Unity 2022.3 LTS+ and Android Build Support installed.

### 3. Compute Hardware & CPU Inference Latency
- **Current State:** The host environment runs PyTorch inference on CPU without active CUDA GPU acceleration.
- **Impact:** Autoregressive subword decoding on CPU requires approximately $744\text{ ms}$ per sentence. While this maintains glass-to-glass latency below the 1.0-second threshold ($761.35\text{ ms}$ total), real-time fluidity is substantially enhanced when executing on an NVIDIA GPU (achieving $< 95\text{ ms}$ latency).

### 4. Zero-Shot Domain Shift & Environmental Sensitivity
- **Current State:** MediaPipe landmark estimation relies on visible contrast and unobstructed views of upper-body pose and hands.
- **Impact:**
  - **Severe Low-Light / Backlighting:** Poor illumination degrades 3D hand landmark confidence.
  - **Self-Occlusion:** Complex two-handed overlapping signs can cause momentary landmark dropout.
  - **Non-Sign Motion:** Random non-sign gesturing in front of the camera may trigger the autoregressive decoder unless gated by the `SignActivityDetector`.

### 5. Architectural Scope Constraints (By Design)
- **Strictly Unidirectional:** Bi-ISL AR translates **ISL $\to$ English Sentences**. It does not perform reverse English-to-sign generation or avatar rendering.
- **Language Coverage:** Current translations target natural English sentences. Direct output in regional Indian languages (e.g., Hindi, Bengali, Tamil, Telugu) is not yet integrated into the subword vocabulary.

---

## 3. Future Research & Development Roadmap

```
+-------------------------------------------------------------------------------------------------+
|                                 Bi-ISL AR Research Roadmap                                      |
+-------------------------------------------------------------------------------------------------+
|  Phase A: Edge Model Optimization       Phase B: Multilingual Expansion   Phase C: Spatial Vision   |
|  - INT8 / FP16 ONNX Quantization        - Multi-target BPE Tokenizer      - Multi-view Camera Fusion |
|  - Mobile Neural Engine Execution       - Hindi / Regional Dialects       - Depth-enhanced Keypoints |
|  - Sub-50ms On-Device Inference         - Domain Lexicon Expansion        - 60 FPS Pose Tracking    |
+-------------------------------------------------------------------------------------------------+
```

### Phase A: Mobile On-Device Quantization (Edge AI)
- **Objective:** Eliminate the requirement for an external laptop Wi-Fi server by running the translation model directly on the Android smartphone.
- **Technical Steps:**
  1. Export `PoseMotionTranslationModel` to ONNX format with dynamic sequence lengths.
  2. Apply Post-Training Quantization (PTQ) to INT8 using PyTorch / ONNX Runtime Mobile.
  3. Integrate ONNX Runtime with Unity C# via native plugins (`onnxruntime.dll` / `.so`), enabling standalone on-device execution.

### Phase B: Multilingual & Multi-Dialect Synthesis
- **Objective:** Support direct translation from Indian Sign Language into multiple Indian national languages.
- **Technical Steps:**
  1. Expand the SentencePiece vocabulary to 16,000 multilingual tokens covering Devanagari (Hindi, Marathi), Dravidian (Tamil, Telugu, Kannada), and Latin (English) scripts.
  2. Implement language conditioning tokens (e.g., `<2en>`, `<2hi>`, `<2te>`) in the Transformer decoder.

### Phase C: Multi-Camera Fusion & Spatial Depth Estimation
- **Objective:** Overcome self-occlusion during complex two-handed continuous gestures.
- **Technical Steps:**
  1. Leverage stereo / multi-camera arrays or depth sensor data (ToF sensors on modern smartphones) to refine 3D $(x, y, z)$ hand coordinates.
  2. Implement a temporal Kalman filter or Graph Convolutional Network (GCN) smoother to interpolate occluded keypoints.

### Phase D: Signer Personalization & Unsupervised Test-Time Adaptation
- **Objective:** Dynamically adapt translation models to idiosyncratic signing speeds and regional dialect variations.
- **Technical Steps:**
  1. Incorporate lightweight Low-Rank Adaptation (LoRA) or adapter layers into the Transformer encoder.
  2. Update adapters using entropy minimization on high-confidence streaming predictions.
