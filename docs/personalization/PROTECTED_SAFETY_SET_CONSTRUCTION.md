# Protected Reference Safety Set Protocol & Construction Spec (Prompt 44)

## Core Design & Safety Isolation Principles

During Unsupervised Group/Signer Adaptation (UGSA), lightweight signer adapters iteratively update model representations on unlabelled live stream inputs. To prevent **catastrophic forgetting** or drift from general ISL linguistic domain rules, UGSA incorporates a **Protected Reference Safety Set**.

---

## Technical Specifications

1. **Size & Composition:**
   - Fixed small sample size: $N = 50$ representative calibration phrases.
   - Sampled across core general ISL domains (greeting, emergency, medical, spatial, numeric).

2. **Strict Operational Constraints:**
   - **Isolation Rule:** **NEVER used for normal online adapter weight updates.**
   - **Zero Test Contamination Guarantee:** Validated via automated ID disjointness checks (`audit_non_contamination`).
   - **Integrity & Versioning:** Tracked via deterministic SHA256 manifest hash and semantic version IDs (`v1.0.0`).

3. **Pre/Post Adaptation Degradation Audit:**
   - Before accepting an online adapter gradient step, model performance is evaluated on the protected safety set.
   - **Degradation Formula:**
     $$\Delta_{\text{safety}} = \frac{\text{BLEU}_{\text{pre}} - \text{BLEU}_{\text{post}}}{\text{BLEU}_{\text{pre}}} \times 100\%$$
   - **Emergency Rollback Threshold:** If $\Delta_{\text{safety}} > 5.0\%$, the online adapter update is rejected, and weights are automatically rolled back to the prior stable checkpoint.
