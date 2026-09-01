# ADR-003: Gloss-Free vs. Gloss-Assisted Continuous SLT

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Modeling & NLP Team  
**Traceability:** `O1`, `E1`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#5-scientific-novelty-boundaries--prior-work-contrast)

---

## Context

Continuous Sign Language Translation (SLT) converts continuous sign video sequences into spoken language sentences (e.g., English text). Literature split between two paradigms:

1.  **Gloss-Assisted SLT (CSLR $\rightarrow$ SLT):** Uses intermediate sign glosses (written tokens representing individual signs in sign word order) as an intermediate supervision target (Continuous Sign Language Recognition followed by Translation).
2.  **Gloss-Free Direct SLT (SignVideo $\rightarrow$ Text):** Directly maps spatiotemporal visual representations to natural language text sequences end-to-end without requiring intermediate gloss annotations.

Public ISL datasets (`ISLTranslate`, `iSign`) contain sentence-level English translations, but gloss annotations are either absent, sparse, or inconsistent across signers and regions.

---

## Decision

We decide to adopt a **primary Gloss-Free Direct Sequence-to-Sequence Translation architecture** with an **optional auxiliary CTC gloss-alignment head** when gloss annotations are available (`iSign`).

*   The main translation decoder translates spatiotemporal visual features directly into English text.
*   An auxiliary Connectionist Temporal Classification (CTC) loss head can be attached to the visual temporal encoder during pretraining if gloss labels exist, but is not required for inference.

---

## Alternatives Considered

1.  **Strict Two-Stage Cascaded SLT (Video $\rightarrow$ Gloss $\rightarrow$ Text):** Explicitly decoding glosses first, then translating glosses to text.  
    *Rejected:* Gloss decoding errors propagate directly into text generation; lacks gloss datasets for full ISL benchmarks.
2.  **Pure Gloss-Free with Zero Intermediate Losses:** Completely ignoring gloss annotations across all datasets.  
    *Rejected:* Discards helpful temporal alignment supervision available in datasets like `iSign`.

---

## Advantages

*   Eliminates dependency on dense frame-level gloss annotations across all ISL datasets.
*   Avoids error compounding from misclassified intermediate gloss tokens.
*   Aligns with modern end-to-end vision-language translation literature.

---

## Risks

*   Direct end-to-end sequence translation requires sufficient training sample volume to learn spatiotemporal alignments without intermediate gloss supervision.

---

## Consequences

*   `src/models/baseline_slt.py` implements end-to-end attention decoding over temporal visual representations.
*   `src/models/losses.py` provides optional auxiliary CTC loss modules for dataset subsets containing gloss targets.

---

## Revisit Conditions

Revisit if a fully standardized, densely annotated ISL gloss corpus becomes available across all major benchmarks.
