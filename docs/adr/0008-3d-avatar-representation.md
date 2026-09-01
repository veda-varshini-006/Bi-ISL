# ADR-008: 3D Avatar Representation & Procedural Animation Engine

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Avatar & Graphics Team  
**Traceability:** `O6`, `RQ6`, `E9`, `E10`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#2-formal-research-objectives-o1o8)

---

## Context

To complete the bidirectional communication loop, reverse translation outputs must be rendered visually for Deaf and Hard-of-Hearing (DHH) users. Text subtitles alone are insufficient for users whose primary language is ISL.

Rendering sign language requires visualizing:
1.  **Manual Articulations:** Complex finger configurations, hand orientation, and 3D signing trajectories.
2.  **Non-Manual Markers:** Facial expressions (eyebrows raised/lowered), mouthing/mouth gestures, head tilts, and torso shifts essential for ISL grammar (e.g., question marking).
3.  **Smooth Motion Transitions:** Coarticulation transitions between consecutive signs.

---

## Decision

We decide to implement a **rigged 3D avatar animation engine** driven by a structured ISL Intermediate Representation (IR):

1.  **Skeletal Rig & Facial Blendshapes:** Uses a standard humanoid 3D mesh equipped with a high-fidelity hand skeleton (5 fingers, 3 joints each) and facial blendshapes for grammatical non-manual expressions.
2.  **Procedural Coarticulation:** Applies keyframe interpolation with Inverse Kinematics (IK) and cubic spline blending to smooth transitions between consecutive gloss animations.
3.  **Cross-Platform Delivery:** Rendered via WebGL / Unity runtime engine integrated into the mobile application shell.

---

## Alternatives Considered

1.  **Pre-Rendered Video Clip Stitching:** Concatenating recorded 2D video clips of a human signer.  
    *Rejected:* Large asset size (gigabytes), harsh visual jumps at clip boundaries, incapable of dynamic non-manual composition.
2.  **Real-Time Neural Video Synthesis (NeRF / Diffusion):** Generating photorealistic video frames on the fly.  
    *Rejected:* Exceeds mobile GPU capability; cannot maintain sub-200ms real-time rendering.
3.  **2D Vector Graphic Animations:** Simplified 2D cartoon avatars.  
    *Rejected:* Cannot render 3D hand depth and complex orientation changes accurately.

---

## Advantages

*   Lightweight asset footprint ($< 30\text{ MB}$ 3D model and motion library) executable locally on mobile devices.
*   Enables independent, parametric control over manual signs and non-manual facial blendshapes.
*   Produces smooth, continuous sign transitions without video splicing artifacts.

---

## Risks

*   Procedural motion interpolation can appear rigid or unnatural ("uncanny valley") if joint acceleration curves are uncalibrated.

---

## Consequences

*   `src/avatar/avatar_renderer.py` manages motion playback, blendshape targets, and timing signals.
*   `src/avatar/assets/` stores 3D avatar meshes, skeletal rigs, and sign motion definitions.
*   Experiment E10 evaluates avatar readability with DHH participants.

---

## Revisit Conditions

Revisit if real-time neural avatar video synthesis achieves sub-50ms execution on commodity mobile GPUs.
