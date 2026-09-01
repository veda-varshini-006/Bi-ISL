# Combined SBDS Context Gating + UGSA Personalization Architecture (Prompt 51)

## Architectural Synthesis & Isolation Principles

The **Combined SBDS + UGSA Pipeline** (`CombinedSBDSUGSAPipeline`) couples dialogue context gating (Phase 4) and online signer adaptation (Phase 5) into a unified end-to-end Sign Language Translation system without mutating frozen base visual/translation model parameters.

---

## Strict Isolation Guarantees

1. **Context Gating Isolation Guarantee:**
   - SBDS dialogue context vector $c_t$ and reliability gate score $\alpha_t$ affect ONLY the final translation decoder logits.
   - UGSA adaptation confidence metrics ($p_t, q_t$) are computed strictly on raw un-gated visual evidence representations ($h_{\text{visual}}$) to prevent context feedback loops from falsely inflating adaptation confidence.

2. **Adaptation Isolation Guarantee:**
   - Online UGSA gradient steps ($\theta_u \leftarrow \theta_u - \eta \nabla \mathcal{L}$) modify ONLY local `SignerAdapter` parameters.
   - Dialogue state objects (`SharedBidirectionalDialogueState`) remain versioned and immutable; adaptation gradient steps never corrupt dialogue state.

3. **Independent Disabling Control Flags:**
   - `enable_sbds: bool`: Toggles SBDS context gating ON/OFF.
   - `enable_ugsa: bool`: Toggles UGSA signer adaptation ON/OFF.

---

## 2x2 Modular Abstraction Grid

| Modular Configuration | `enable_sbds` | `enable_ugsa` | Description |
| :--- | :---: | :---: | :--- |
| **Config A (Generic Baseline)** | `False` | `False` | No context, no signer adaptation |
| **Config B (Context Only)** | `True` | `False` | SBDS context gating active, no adaptation |
| **Config C (UGSA Only)** | `False` | `True` | UGSA signer adaptation active, no context |
| **Config D (Combined System)** | `True` | `True` | SBDS context gating + UGSA active |
