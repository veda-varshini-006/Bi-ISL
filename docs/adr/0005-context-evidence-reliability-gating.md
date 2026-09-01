# ADR-005: Context-Evidence Reliability Gating Mechanism

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Modeling & Safety Team  
**Traceability:** `O3`, `RQ2`, `H2`, `E2`, `E3`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#2-formal-research-objectives-o1o8)

---

## Context

Integrating conversational context into continuous SLT decoders introduces a critical safety risk: **context overriding visual evidence**. When dialogue history contains entities or attributes that conflict with current visual signing input (e.g., irrelevant prior topics, misheard ASR outputs, or adversarial contradictory turns), conventional context models generate visually unsupported sentences (hallucinations).

A publishable system must dynamically balance visual evidence against conversational context to ensure translation faithfulness.

---

## Decision

We decide to develop a **Context-Evidence Reliability Gate** ($G_t \in [0, 1]$) that computes the alignment between visual spatiotemporal representations and dialogue context state ($S_t$).

$$\mathbf{h}_{\text{combined}} = \mathbf{h}_{\text{visual}} + G_t \cdot \mathbf{h}_{\text{context}}$$

The gate value $G_t$ is computed via:
1.  **Visual Confidence Estimator:** Computes spatiotemporal encoder certainty on current video frames.
2.  **Context-Visual Similarity Score:** Measures semantic consistency between visual frame representations and dialogue state entities.
3.  **Dynamic Down-Weighting:** When visual confidence is high or context contradicts visual evidence, $G_t \rightarrow 0$, enforcing pure visual translation. When visual features are ambiguous and context is consistent, $G_t \rightarrow 1$.

---

## Alternatives Considered

1.  **Static Fixed Context Weight ($G_t = 0.5$):** Applying a fixed constant context weight across all samples.  
    *Rejected:* Fails to protect against contradictory context attacks (E3).
2.  **Ungated Soft Cross-Attention:** Relying solely on standard Transformer cross-attention mechanisms.  
    *Rejected:* Decoders routinely over-attend to language context over noisy visual keys.
3.  **Post-Hoc Verification Filter:** Generating text with context and filtering outputs after decoding.  
    *Rejected:* Computationally wasteful and creates decoding latency bottlenecks.

---

## Advantages

*   Dramatically reduces Unsupported Slot Rate (USR) and semantic corruption under misleading context attacks (E3).
*   Maintains high context-dependent BLEU-4 gains on clean dialogue sequences (E2).
*   Provides an explicit reliability score that can trigger abstention / low-confidence warnings.

---

## Risks

*   If the reliability gate is miscalibrated, it may erroneously suppress helpful context when visual frames are degraded by lighting or occlusion.

---

## Consequences

*   `src/models/reliability_gate.py` implements the gating module and confidence metrics.
*   Experiment E3 explicitly evaluates USR and hallucination rate under contradictory history injections.

---

## Revisit Conditions

Revisit if visual encoders achieve perfect unambiguous feature extraction across all lighting and occlusion conditions.
