# ADR-009: Target Mobile Inference Framework for Android

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Mobile & Edge Deployment Team  
**Traceability:** `O7`, `RQ5`, `H5`, `E8`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#73-system--mobile-deployment-metrics)

---

## Context

The mobile deployment prototype (O7) must execute locally on mid-range Android smartphones to ensure offline availability, privacy protection, and sub-200ms turn latency.

Selecting a mobile machine learning inference runtime requires evaluating:
*   PyTorch model export compatibility (tracing/graph conversion).
*   Post-training quantization support (INT8 / FP16).
*   Hardware acceleration backend integration (Android NNAPI, Vulkan, OpenCL, NPU).
*   Runtime binary footprint and memory overhead.

---

## Decision

We decide to select **ONNX Runtime Mobile / ExecuTorch** as the target inference engine for Android deployment:

1.  **Export Path:** High-capacity PyTorch research models are exported to Open Neural Network Exchange (ONNX) format or ExecuTorch IR graph format.
2.  **Quantization:** Models undergo INT8 static/dynamic quantization and FP16 operator conversion via ONNX Runtime / ExecuTorch quantization tooling.
3.  **Mobile Execution:** Executed via native C++ ONNX Runtime / ExecuTorch bindings inside the Android application container (`src/deployment/android/`), utilizing NNAPI / Vulkan hardware acceleration backends.

---

## Alternatives Considered

1.  **TensorFlow Lite (TFLite):** Exporting PyTorch models to TFLite via ONNX/Keras intermediate tools.  
    *Rejected:* Multi-step cross-framework conversion frequently breaks complex custom attention operators.
2.  **Legacy PyTorch Mobile:** Using the original `torchscript` mobile runtime.  
    *Rejected:* PyTorch Mobile is deprecated in favor of ExecuTorch; lacks active optimization updates.
3.  **Server-Side Cloud API Execution:** Sending video streams to remote cloud servers.  
    *Rejected:* Violates offline execution requirements and introduces network latency unpredictability.

---

## Advantages

*   Direct, robust export path from PyTorch research codebase (`src/models/`).
*   Advanced INT8 quantization and hardware acceleration across diverse Android NPU/GPU chipsets.
*   Compact runtime binary size ($< 15\text{ MB}$) and low RAM overhead.

---

## Risks

*   Custom spatiotemporal attention layers or unusual tensor operations may require custom C++ ONNX operator implementations.

---

## Consequences

*   `src/deployment/mobile_export.py` handles PyTorch-to-ONNX/ExecuTorch export, quantization, and numerical equivalence verification.
*   Experiment E8 benchmarks $p50/p95$ latency, peak RAM, storage footprint, and battery consumption on target Android hardware.

---

## Revisit Conditions

Revisit if ExecuTorch achieves complete feature parity with PyTorch mobile backends across all Android vendor NPUs.
