# Naive Signer Fine-Tuning Controlled Baseline Specifications (Prompt 48)

## Purpose & Baseline Role

To validate whether UGSA's confidence gating ($g_t$) and transactional safety rollback mechanisms are methodologically necessary, we implement **Naive Signer Fine-Tuning** (`NaiveSignerFineTuningBaseline`).

---

## Controlled Experimental Alignment

To isolate the specific impact of safety gating and rollback mechanisms:
- **Same Adaptation Samples:** Processed on the exact same sequence batches as UGSA.
- **Same Computational Budget:** Same learning rate ($\eta = 10^{-4}$), optimizer (AdamW), and gradient step count.
- **NO Confidence Gating:** $g_t \equiv 1$ (all samples trigger gradient updates regardless of confidence or noise).
- **NO Safety Rollback:** Performance on general reference safety sets is unmonitored; bad updates commit permanently.

---

## Comparative Safety Feature Matrix

| Feature | Proposed UGSA System | Naive Fine-Tuning Baseline |
| :--- | :---: | :---: |
| **Adaptation Samples** | Stream Inputs | Stream Inputs (Identical) |
| **Learning Rate ($\eta$)** | $10^{-4}$ | $10^{-4}$ (Identical) |
| **Confidence Gate ($p_t, q_t$)** | Enabled ($\tau_p=0.85, \tau_q=0.75$) | Bypassed / Disabled |
| **Protected Set Monitoring** | Enabled | Disabled |
| **Atomic Safety Rollback** | Active ($\Delta > 5\% \implies \text{Rollback}$) | None (Permanent Commit) |
