# Established Sign-Language TTA Baseline Reproduction Report (Prompt 49)

## Academic Citation & Prior-Art Context

- **Baseline Name:** SAME (Sign-Language Adaptation via Feature Statistics & Entropy Minimization)
- **Primary Paradigm:** Test-Time Adaptation (TTA) via Softmax Entropy Minimization + Source Feature Alignment.

---

## Honest Reproduction & Architectural Compatibility Audit

### Original SAME Formulation vs Bi-ISL Sequence Landmark Architecture

1. **Original SAME Constraints:**
   - Designed for 2D CNN (I3D / ResNet) frame-level isolated sign gloss classifiers.
   - Updates Batch Normalization affine parameters $(\gamma, \beta)$ continuously during test-time streaming.

2. **Bi-ISL Architectural Differences & Adaptation:**
   - **Difference:** Bi-ISL uses 3D Mediapipe Landmark coordinates with LayerNorm and 1D TCN + Autoregressive Decoder.
   - **Mathematically Justified Adaptation:**
     - Applies SAME entropy minimization loss on sequence decoder output logits:
       $$\mathcal{L}_{\text{entropy}} = -\frac{1}{T} \sum_{t=1}^T \sum_{v \in \mathcal{V}} p(v_t) \log p(v_t)$$
     - Aligns 1D temporal feature statistics $(\mu_{\text{stream}}, \sigma^2_{\text{stream}})$ to source reference distributions $(\mu_{\text{ref}}, \sigma^2_{\text{ref}})$:
       $$\mathcal{L}_{\text{align}} = \|\mu_{\text{stream}} - \mu_{\text{ref}}\|_2^2 + \|\sigma^2_{\text{stream}} - \sigma^2_{\text{ref}}\|_2^2$$
     - Updates active `SignerAdapter` parameters.

---

## Objective Function & Hyperparameters

$$\mathcal{L}_{\text{SAME}} = \mathcal{L}_{\text{entropy}} + \gamma \mathcal{L}_{\text{align}}$$

- **Learning Rate ($\eta$):** $1 \times 10^{-4}$
- **Feature Alignment Weight ($\gamma$):** `0.1`
- **Optimizer:** AdamW
