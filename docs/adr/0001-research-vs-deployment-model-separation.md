# ADR-001: Research vs. Deployment Model Separation

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Core Architecture Team  
**Traceability:** `O1`, `O7`, `E1`, `E8`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#1-project-aim)

---

## Context

Deep learning models for continuous Sign Language Translation (SLT) require high-capacity spatiotemporal backbones (e.g., Video Transformers, heavy 3D-CNNs, deep spatial cross-attention layers) to maximize sentence BLEU-4 and benchmark performance. However, deploying such heavy models directly onto mobile devices (Android) causes severe out-of-memory (OOM) failures, excessive thermal throttling, battery drain, and end-to-end latency exceeding $1000\text{ ms}$, rendering real-time communication impossible.

Attempting to train a single lightweight model to serve both research benchmarking and mobile deployment results in a compromised architecture: it fails to compete with state-of-the-art literature on benchmark accuracy while still remaining too heavy for real-time mobile execution.

---

## Decision

We decide to **decouple the primary research model from the mobile deployment model**:

1.  **Research Model Pipeline (`src/models/`):** Optimized for maximum translation quality, context-evidence gating, signer adaptation research, and benchmark comparability (E1–E7). Uses high-capacity PyTorch architectures evaluated on workstation/GPU hardware.
2.  **Mobile Deployment Pipeline (`src/deployment/`):** Derived from the research baseline through systematic post-training optimization, operator fusion, FP16/INT8 quantization, and optional knowledge distillation (E8). Optimized specifically for target mobile hardware (Android NPU/GPU/CPU).

Both pipelines share identical input data schemas and dialogue state data structures to preserve behavioral alignment.

---

## Alternatives Considered

1.  **Monolithic Single Model:** Using one model for both research experiments and mobile execution.  
    *Rejected:* Severely limits benchmark accuracy or causes mobile execution failure.
2.  **Mobile-Only Model Trained from Scratch:** Training a lightweight mobile architecture from scratch without a high-capacity research baseline.  
    *Rejected:* Lacks state-of-the-art representation capacity, leading to poor translation quality.
3.  **Cloud-Only API Execution:** Streaming video to a cloud server for inference.  
    *Rejected:* Violates offline availability, introduces network latency variability, and compromises user privacy.

---

## Advantages

*   Preserves state-of-the-art benchmark competitiveness for academic publications.
*   Enables realistic mobile performance benchmarking ($p95 < 200\text{ ms}$) without degrading research model accuracy.
*   Allows independent iteration on deep spatiotemporal architectures and mobile quantization techniques.

---

## Risks

*   Potential accuracy degradation during post-training quantization or model distillation.
*   Maintenance overhead of maintaining dual model export scripts and runtime backends.

---

## Consequences

*   High-capacity PyTorch research models live under `src/models/`.
*   Quantization, ONNX/ExecuTorch export, and mobile execution code live under `src/deployment/`.
*   Experiment E8 explicitly benchmarks and quantifies the accuracy-latency tradeoff between research and mobile model variants.

---

## Revisit Conditions

This decision will be revisited if edge NPU accelerators evolve to execute heavy spatiotemporal video transformers natively at $< 50\text{ ms}$ latency without requiring model compression or quantization.
