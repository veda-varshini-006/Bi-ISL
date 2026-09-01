# ADR-006: Uncertainty-Gated Signer Adaptation (UGSA) & Protected Rollback

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Personalization & Reliability Team  
**Traceability:** `O4`, `RQ3`, `H3`, `E4`, `E5`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#2-formal-research-objectives-o1o8)

---

## Context

Signers exhibit individual variations in signing speed, physical proportion, gesture trajectory, spatial signing space, and regional dialect vocabulary. Generic global models often perform poorly on non-standard signers.

However, applying online Test-Time Adaptation (TTA) using unverified pseudo-labels introduces **adaptation poisoning**: noisy pseudo-labels cause model weights to drift, leading to catastrophic degradation on previously accurate signers or protected baseline datasets.

---

## Decision

We decide to implement **Uncertainty-Gated Signer Adaptation (UGSA)** with **Protected-Set Rollback Logic**:

1.  **Uncertainty-Gated Parameter Updates:** Online parameter updates (via lightweight adapter modules or LayerNorm tuning) are executed *only* when prediction uncertainty (entropy/variance) falls below a strict threshold $\tau_{\text{uncert}}$. Low-confidence or noisy frames are excluded from adaptation.
2.  **Protected Baseline Verification:** After every $N$ adaptation steps, the adapted model is evaluated against a fixed held-out protected baseline dataset.
3.  **Automatic Rollback Execution:** If protected-set BLEU-4 degrades by more than $\Delta_{\text{max}} = 2.0\%$, the system automatically rolls back model weights to the last verified checkpoint ($W_{\text{safe}}$) and resets adaptation learning rates.

---

## Alternatives Considered

1.  **Static Generic Model:** No online personalization or test-time adaptation.  
    *Rejected:* Caps accuracy for non-standard signers with unique signing styles.
2.  **Naive Unsafe Test-Time Adaptation (P1):** Continuous gradient updates on all pseudo-labels without gating or checks.  
    *Rejected:* Vulnerable to adaptation poisoning and catastrophic model drift (E5).
3.  **Supervised Signer Fine-Tuning:** Requiring ground-truth annotations for every user.  
    *Rejected:* Impractical for real-world interactive deployment.

---

## Advantages

*   Improves median and average per-signer translation performance across target signers (E4).
*   Enforces a strict upper bound on worst-signer performance degradation ($\le 2.0\%$).
*   Guarantees safety against noisy pseudo-labels and adversarial feedback inputs (E5).

---

## Risks

*   Strict uncertainty thresholds may slow down adaptation convergence for signers with highly atypical gestures.

---

## Consequences

*   `src/adaptation/ugsa_adapter.py` manages parameter updates and uncertainty estimation.
*   `src/adaptation/rollback.py` implements protected-set tracking and checkpoint rollback triggers.

---

## Revisit Conditions

Revisit if foundation models achieve universal zero-shot invariance across all regional ISL dialects without test-time adaptation.
