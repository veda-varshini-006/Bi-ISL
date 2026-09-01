# Bi-ISL Confidence Calibration Specifications & Metrics (Prompt 42)

## Overview & Strict Principles

In Sign Language Translation (SLT), raw softmax probability distributions from neural decoders are inherently overconfident and poorly calibrated. Treating raw softmax confidence as an automated indicator of translation correctness leads to unwarranted online adapter updates on noisy or hallucinated outputs.

**Strict Mandate:** **Never treat raw softmax confidence as automatically calibrated.**

---

## Confidence Calibration Architecture (`ConfidenceCalibrator`)

1. **Post-Hoc Temperature Scaling:**
   $$p_{i, T} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - Parameter $T > 0$ is optimized post-hoc on validation logits to minimize negative log-likelihood.

2. **Uncertainty Metrics Evaluated:**
   - **Predictive Entropy $H(p)$:**
     $$H(p) = -\sum_{v \in \mathcal{V}} p(v) \log p(v)$$
   - **Sequence Agreement Rate:** Measures consensus consistency across $K$ stochastic sampling decodings.

3. **Formal Calibration Metrics:**
   - **Expected Calibration Error (ECE):**
     $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
   - **Brier Score:**
     $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$$
