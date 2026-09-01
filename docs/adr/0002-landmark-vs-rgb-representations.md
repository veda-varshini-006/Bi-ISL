# ADR-002: Landmark-Based vs. RGB/Video Representations

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Vision & Feature Extraction Team  
**Traceability:** `O1`, `O7`, `E1`, `E8`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#6-experimental-program-e0e10-matrix)

---

## Context

Indian Sign Language (ISL) conveys information through manual articulations (hand shapes, orientation, trajectory), upper-body pose, and non-manual expressions (facial expressions, mouthings, head tilts).

Capturing input video requires selecting an input representation:
1.  **Raw RGB Video Frames:** Carry rich texture and visual detail, but suffer from heavy background/lighting domain shift, signer appearance bias, high storage/bandwidth requirements, and heavy spatiotemporal compute overhead.
2.  **Landmark Keypoints (2D/3D Pose, Hand, Face):** Provide compact, privacy-preserving geometric representations invariant to background and lighting shifts, but risk losing subtle non-manual cues or suffering keypoint tracking failures under fast motion / self-occlusion.

---

## Decision

We decide to adopt a **hybrid multimodal feature representation strategy**:

1.  **Research Baseline:** Supports dual feature streams: 2D/3D landmark keypoints ($K_t$) extracted via MediaPipe/OpenPose and learned spatiotemporal RGB features ($V_t$) extracted via pretrained backbones (e.g., I3D / Video-Swin).
2.  **Mobile Deployment Prototype:** Uses 2D/3D landmark keypoints ($K_t$) as the primary lightweight input stream extracted on-device via MediaPipe.

---

## Alternatives Considered

1.  **Pure Landmark Keypoints Only:** Discarding RGB frames entirely across all research baselines.  
    *Rejected:* May cap peak translation accuracy on datasets where facial expression texture is critical.
2.  **Pure Raw RGB Frames Only:** Processing raw video frames through a end-to-end 3D-CNN on mobile devices.  
    *Rejected:* Exceeds mobile memory and latency budgets; highly sensitive to background lighting variations.
3.  **Depth / IR Sensor Inputs:** Requiring depth sensors or specialized hardware.  
    *Rejected:* Prevents deployment on standard commodity smartphones.

---

## Advantages

*   Landmarks eliminate visual domain shift (backgrounds, lighting, clothing), improving cross-domain generalization (E7).
*   Enables real-time landmark extraction on commodity Android smartphones.
*   Preserves user privacy by avoiding raw video storage or transmission.

---

## Risks

*   Keypoint tracking jitter or loss during rapid hand movements or partial hand-face occlusions.
*   Coarse facial keypoints may incompletely capture subtle ISL non-manual grammatical markers.

---

## Consequences

*   The visual preprocessing module (`src/vision/`) must implement dual adaptors: keypoint normalization (`src/vision/landmarks.py`) and RGB feature extraction (`src/vision/rgb_extractor.py`).
*   Data pipeline caching must store pre-extracted keypoint matrices for fast epoch iteration.

---

## Revisit Conditions

Revisit if lightweight on-device vision backbones achieve faster execution than MediaPipe landmark extraction while maintaining zero background bias.
