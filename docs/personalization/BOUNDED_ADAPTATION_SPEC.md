# Bounded Signer Adaptation Specifications (Prompt 46)

## Regularized Objective Formulation

Unsupervised Signer Adaptation (UGSA) optimizes a regularized objective to bound parameter drift from initial base weights $\theta_0$:

$$\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{task}} + \lambda \|\theta_u - \theta_0\|_2^2$$

where:
- $\mathcal{L}_{\text{task}}$: Unsupervised self-training loss (pseudo-label cross-entropy).
- $\theta_u$: Current trainable adapter weights.
- $\theta_0$: Frozen initial adapter baseline state.
- $\lambda$: L2 regularization coefficient ($\lambda = 0.01$).

---

## Hard Operational Constraints

1. **Max Gradient Steps ($N_{\text{max\_steps}}$):** $N \le 5$ steps per session.
2. **Learning Rate Upper Bound ($\eta$):** $\eta \le 1 \times 10^{-4}$.
3. **Hard Distance Ball Radius ($R_{\text{max}}$):** $\|\theta_u - \theta_0\|_2 \le 0.50$.
4. **Buffer Memory Limit ($K_{\text{max}}$):** Retains at most $K = 100$ calibration samples.
5. **Adapter Parameter Limit:** $< 50,000$ trainable parameters per signer.
