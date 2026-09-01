# ADR-004: Shared Bidirectional Dialogue State (SBDS)

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Dialogue & System Architecture Team  
**Traceability:** `O2`, `RQ4`, `H4`, `E2`, `E6`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#2-formal-research-objectives-o1o8)

---

## Context

Conversational sign language systems must support alternating multi-turn interactions (ISL user signing $\rightarrow$ English translation $\rightarrow$ Hearing user speaking/typing $\rightarrow$ ISL avatar response). Standard translation systems treat each turn independently, resulting in misresolved referents, dropped entities, and inconsistent conversational intent across turns.

Conversely, passing unstructured raw text transcripts of prior turns into language decoders leads to severe visual context hallucination (decoders generating fluent text backed by historical transcript rather than visual evidence).

---

## Decision

We decide to implement a compact, structured **Shared Bidirectional Dialogue State (SBDS)** represented as:

$$S_t = \{E_t, I_t, R_t, T_t, C_t, Y_{t-1}\}$$

Where:
*   $E_t$: Active entities extracted across turns.
*   $I_t$: Active conversational intent (e.g., query, confirmation, direction).
*   $R_t$: Unresolved referents and pronouns.
*   $T_t$: Spatial/temporal grounding attributes.
*   $C_t$: Confidence metadata per extracted state element.
*   $Y_{t-1}$: Previous confirmed turn text / structured response.

This structured state is shared between the forward continuous ISL translation decoder and the reverse English-to-ISL structured generator.

---

## Alternatives Considered

1.  **Uncoupled Single-Turn Translation:** Processing each turn with zero historical memory.  
    *Rejected:* Fails to resolve dialogue referents and context-dependent sign ambiguities.
2.  **Raw Transcript Concatenation:** Concatenating previous text strings directly into input prompts.  
    *Rejected:* Induces high hallucination rates when historical text conflicts with current video input.
3.  **Dense Neural Memory Embeddings (RAG/Vector DB):** Maintaining uninterpretable vector memory vectors.  
    *Rejected:* Lacks interpretability, expensive for on-device state management, difficult to inspect during safety failures.

---

## Advantages

*   Provides structured, interpretable conversational memory across multi-turn interactions.
*   Eliminates unstructured text hallucination channels.
*   Lightweight footprint suitable for on-device mobile state tracking ($< 1\text{ MB}$).

---

## Risks

*   Complex dialogue turns with highly nested clauses may be difficult to map into structured state slots.

---

## Consequences

*   `src/dialogue/sbds_manager.py` manages entity extraction, state updates, confidence tracking, and turn decay.
*   Both forward (`src/models/context_gate.py`) and reverse (`src/reverse/structured_gen.py`) pipelines interface directly with SBDS.

---

## Revisit Conditions

Revisit if open-domain multi-turn dialogue complexity requires dynamic graph-structured memory representations.
