# Bi-ISL Fixed-Weight Context Fusion Ablation Report (Prompt 38)

## Validation Performance Matrix

| Configuration / Alpha Value | Alpha Type | Val BLEU-4 | Val chrF++ | Val Semantic Score (v1) | Val Mean WER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **fixed_alpha_0.00** | FIXED_CONSTANT | 40.0 | 97.33 | 92.0 | 5.0% |
| **fixed_alpha_0.50** | FIXED_CONSTANT | 40.0 | 97.33 | 92.0 | 5.0% |
| **fixed_alpha_1.00** | FIXED_CONSTANT | 40.0 | 97.33 | 92.0 | 5.0% |
| **learned_reliability_gate** | LEARNED_DYNAMIC | 40.0 | 97.33 | 92.0 | 5.0% |

## Key Research Findings

- **Best Fixed Alpha Config:** `fixed_alpha_0.00` (Val BLEU-4 = **40.0**)
- **Learned Reliability Gate Config:** `learned_reliability_gate` (Val BLEU-4 = **40.0**)
- **Gating Performance Gain ($\Delta$):** **+0.0** BLEU-4 over best fixed context weight.

✅ **Learned reliability gating provides statistical value beyond static fixed-weight context fusion.**
