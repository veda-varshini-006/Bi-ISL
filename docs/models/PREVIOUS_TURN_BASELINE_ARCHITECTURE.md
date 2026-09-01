# Bi-ISL Previous-Turn Ungated Context Baseline Architecture (Prompt 37)

## Overview & Prior-Art Context

The **Previous-Turn Baseline** (`PreviousTurnBaseline`) represents standard prior-art contextual Sign Language Translation (SLT). In conventional contextual SLT architectures, previous turn translations or reference text sequences $T_{t-1}$ are directly encoded and added to the visual representations without structured dialogue state tracking or reliability gating.

---

## Architectural Specifications

1. **Text Sequence Encoder (`PreviousTurnTextEncoder`):**
   - **Embedding Lookup:** Maps input token IDs to $d_{\text{model}} = 256$ dimensional embeddings.
   - **Sequence Encoder:** Single-layer unidirectional GRU ($d_{\text{hidden}} = 256$).
   - **Linear Projection:** Projects final GRU hidden state into context vector $c_{t-1} \in \mathbb{R}^{256}$.

2. **Ungated Linear Context Fusion:**
   - **Additive Fusion:**
     $$\tilde{h}_t = h_t + W_{\text{prev}} c_{t-1}$$
   - **No Reliability Gating:** $\alpha_t \equiv 1.0$ (Always un-gated).
   - **No SBDS State:** Ignores structured entity memory, intent domain, spatial referents, or confidence signals.

3. **Parameter Capacity:**
   - **Text Encoder:** ~320K parameters.
   - **Total Model Capacity:** ~1.42M parameters (comparable to SBDS context model).

---

## Methodological Comparison Matrix

| Component | Bi-ISL Proposed Mechanism | Previous-Turn Baseline |
| :--- | :--- | :--- |
| **Context Source** | Structured SBDS State Vector | Raw Unstructured Previous Text ($T_{t-1}$) |
| **Context Gating** | Learned Reliability Gate ($\alpha_t$) | Ungated Additive ($\alpha \equiv 1.0$) |
| **Reliability Signals** | 9-Signal Estimator ($u_t$) | None |
| **Contradiction Robustness** | Dynamic Gate Dampening ($\alpha_t \to 0$) | High Vulnerability to Hallucination |
