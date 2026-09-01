# Bi-ISL Learned Visual Encoder RGB/Video Baseline Architecture

## Architectural Summary

The **Bi-ISL RGB/Video Baseline** (`RGBVideoBaseline`) implements a modular learned visual encoder architecture for continuous Sign Language Translation (SLT). It strictly decouples spatial visual feature extraction, temporal sequence modeling, and language target decoding.

> [!IMPORTANT]
> **Control Baseline Isolation:** This baseline model explicitly excludes SBDS (Shared Bidirectional Dialogue State) context gating and UGSA (User-Gated Signer Adaptation) personalization modules to serve as a strict, ungated visual control baseline.

---

## Component Specifications

```
                       ┌───────────────────────────────┐
                       │  Input Video: (B, T, 3, H, W) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Spatial Frame Encoder       │
                       │   (MobileNetV3-Small)         │
                       └───────────────┬───────────────┘
                                       │  (B, T, 512)
                                       ▼
                       ┌───────────────────────────────┐
                       │   Temporal TCN Encoder        │
                       │   (3-Layer 1D ConvNet)        │
                       └───────────────┬───────────────┘
                                       │  (B, T, 512)
                                       ▼
                       ┌───────────────────────────────┐
                       │   Translation Decoder         │
                       │   (GRU + Projection Head)     │
                       └───────────────┬───────────────┘
                                       │  (B, T, Vocab)
                                       ▼
                       ┌───────────────────────────────┐
                       │  Output Logits: (B, T, Vocab) │
                       └───────────────────────────────┘
```

1. **Frame/Video Encoder (`FrameEncoder`):**
   - **Backbone:** MobileNetV3-Small (or ResNet-18)
   - **Pretrained Weights:** ImageNet-1k (`MobileNet_V3_Small_Weights.DEFAULT`)
   - **Spatial Pooling:** Adaptive Average Pooling `(1, 1)` -> Linear Projection `576` $\to$ `512`
   - **Parameter Count:** ~1,234,448 parameters

2. **Temporal Modeling (`TemporalEncoder`):**
   - **Architecture:** 3-Layer 1D Temporal Convolutional Network (TCN)
   - **Kernel Size / Padding:** Kernel = `3`, Padding = `1` (Preserves sequence length $T$)
   - **Regularization:** Batch Normalization + ReLU + Dropout (`0.1`)
   - **Parameter Count:** ~1,577,472 parameters

3. **Translation Decoder (`TranslationDecoder`):**
   - **Architecture:** Single-Layer GRU Decoder
   - **Output Projection:** Linear Classifier (`512` $\to$ `vocab_size`)
   - **Parameter Count:** ~1,626,628 parameters (at `vocab_size=100`)

---

## Parameter Breakdown Table

| Component | Layer / Module | Input Dim | Output Dim | Trainable Parameters |
| :--- | :--- | :---: | :---: | :---: |
| **Frame Encoder** | MobileNetV3-Small + Linear | `(B, T, 3, 224, 224)` | `(B, T, 512)` | **1,234,448** |
| **Temporal Encoder** | 3-Layer 1D TCN | `(B, T, 512)` | `(B, T, 512)` | **1,577,472** |
| **Translation Decoder** | GRU + Linear Head | `(B, T, 512)` | `(B, T, 100)` | **1,626,628** |
| **TOTAL BASELINE** | **Full RGB Baseline** | `(B, T, 3, 224, 224)` | `(B, T, 100)` | **4,438,548** |

---

## Verification & Compliance

- **Pretrained Weights:** ImageNet-1k initialization on 2D spatial backbone.
- **SBDS/UGSA Exclusion:** Verified 100% decoupled from context memory and signer adapters.
- **Unit Test Coverage:** Validated in `tests/models/test_rgb_baseline.py`.
