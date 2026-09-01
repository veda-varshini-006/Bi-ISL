# Transactional Personalization & Safety Rollback Protocol (Prompt 47)

## Transactional State Lifecycle

In Unsupervised Group/Signer Adaptation (UGSA), gradient updates computed on noisy or unlabelled live video streams present a non-zero risk of **catastrophic performance degradation**. To safeguard translation quality, UGSA enforces **Transactional Personalization**.

---

## Protocol Lifecycle Steps

1. **Pre-Update Snapshot:**
   - Before applying an accepted gradient step, the `TransactionalPersonalizationManager` clones the in-memory parameter tensor state dict $\theta_{\text{snapshot}} \leftarrow \theta_u$.

2. **Bounded Gradient Step Execution:**
   - The model parameters are updated via `BoundedSignerUpdater`.

3. **Post-Update Safety Verification:**
   - Model performance is re-evaluated on the `ProtectedSafetySet` to calculate post-update reference score $\text{BLEU}_{\text{post}}$.
   - **Degradation Calculation:**
     $$\Delta_{\text{degrade}} = \frac{\text{BLEU}_{\text{pre}} - \text{BLEU}_{\text{post}}}{\text{BLEU}_{\text{pre}}} \times 100\%$$

4. **Atomic Rollback Condition:**
   - If $\Delta_{\text{degrade}} > \epsilon$ (where $\epsilon = 5.0\%$), the update transaction is aborted and parameters are atomically reverted: $\theta_u \leftarrow \theta_{\text{snapshot}}$.

5. **JSONL Audit Trail Logging:**
   - Transactions append structured records to `artifacts/logs/transactional_adaptation_history.jsonl`.
