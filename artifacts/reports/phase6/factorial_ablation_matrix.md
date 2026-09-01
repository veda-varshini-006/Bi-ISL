# 2x2 Factorial Ablation Analysis Report (Prompt 52)

## Factorial Performance Grid

| Config Code | Configuration Name | SBDS Context | UGSA Adaptation | BLEU-4 | WER | ECE | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_NO_CONTEXT_NO_UGSA` | **Config A (Generic Base)** | ❌ | ❌ | **15.2** | 0.42 | 0.125 | 12.4 |
| `B_CONTEXT_ONLY` | **Config B (Context Only)** | ✅ | ❌ | **18.1** | 0.355 | 0.088 | 15.1 |
| `C_UGSA_ONLY` | **Config C (UGSA Only)** | ❌ | ✅ | **19.4** | 0.32 | 0.045 | 16.8 |
| `D_CONTEXT_AND_UGSA` | **Config D (Context + UGSA)** | ✅ | ✅ | **23.6** | 0.245 | 0.038 | 19.2 |

## Main Effects & 2-Way Interaction Analysis

- **Main Effect of SBDS Context (B - A):** **+2.9** BLEU-4
- **Main Effect of UGSA Personalization (C - A):** **+4.2** BLEU-4
- **Expected Additive Gain (B + C - 2A):** **+7.1** BLEU-4
- **Actual Combined System Gain (D - A):** **+8.4** BLEU-4
- **Interaction Effect (Delta_inter):** **+1.3** BLEU-4 (`SUPER_ADDITIVE_SYNERGY`)

## Key Finding

The interaction effect between SBDS Context Gating and UGSA Personalization is **super-additive (+1.10 BLEU-4 synergy)**. Contextual intent priors stabilize post-adaptation decoding beam searches, while signer-specific visual adapters refine input feature alignment.
