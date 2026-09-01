# ADR-007: Controlled-Domain Reverse ISL Translation Strategy

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Reverse Pipeline & NLP Team  
**Traceability:** `O5`, `RQ6`, `E9`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#5-scientific-novelty-boundaries--prior-work-contrast)

---

## Context

Reverse ISL translation involves converting spoken/written English into sign language output to communicate with Deaf users. English and ISL differ substantially in syntax: English uses Subject-Verb-Object (SVO) ordering with tense inflections, whereas ISL uses Subject-Object-Verb (SOV) / Topic-Comment structures, spatial referents, facial non-manual markers, and specific gloss ordering.

Attempting open-domain text-to-sign motion generation produces unvalidated or grammatically incorrect sign animations that confuse ISL users. Claiming open-domain English-to-ISL translation without complete linguistic validation leads to publication rejection.

---

## Decision

We decide to restrict the reverse English-to-ISL pipeline to **controlled domain contexts** (e.g., healthcare, emergency services, public administration counters):

1.  **Semantic Parsing:** English text/speech input is parsed into a structured Intermediate Representation (IR) encoding intent, entity slots, ISL gloss ordering, and non-manual marker tags.
2.  **Controlled Vocabulary Mapping:** IR tokens map to a dictionary of linguistically validated motion sign primitives and facial blendshape animations.
3.  **Fallback Strategy:** Unmapped out-of-vocabulary (OOV) terms trigger fingerspelling generation or subtitle text overlay rather than synthesizing unvalidated movements.

---

## Alternatives Considered

1.  **Open-Domain Direct Text-to-Pose Generation:** Training a neural generative model for arbitrary text-to-pose synthesis.  
    *Rejected:* Generates fluid-looking motion that lacks ISL grammatical correctness and non-manual clarity.
2.  **Direct Word-for-Word Dictionary Substitution:** Mapping English words directly to sign clips in English SVO word order.  
    *Rejected:* Grammatically inaccurate and confusing for native ISL signers (C2 baseline).

---

## Advantages

*   Guarantees high grammatical correctness and non-manual expression fidelity within target operational domains.
*   Provides a reliable, deterministic translation path for critical accessibility interactions (e.g., medical intake).
*   Facilitates rigorous comprehension validation during DHH user studies (E10).

---

## Risks

*   Restricted vocabulary coverage outside the pre-defined controlled domain templates.

---

## Consequences

*   `src/reverse/structured_gen.py` implements semantic parsing and ISL Intermediate Representation (IR) generation.
*   Experiment E9 explicitly benchmarks controlled IR generation against word-for-word lookup baselines.

---

## Revisit Conditions

Revisit when open-domain text-to-sign pose generation models achieve verified linguistic parity with native ISL signing.
