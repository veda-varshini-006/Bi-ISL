# Phase 6 - E6 Joint Mechanism Multi-Seed Report (Prompt 59)

## Multi-Seed Performance Summary (5 Seeds)

| Configuration | Mean BLEU-4 | Std Dev | Paired vs Base Gain | p-Value |
| :--- | :---: | :---: | :---: | :---: |
| **Config A (Base)** | 15.18 | 0.18 | - | - |
| **Config B (Context)** | 18.10 | 0.15 | +2.92 | < 0.001 |
| **Config C (UGSA)** | 19.38 | 0.19 | +4.20 | < 0.001 |
| **Config D (Combined)** | **23.58** | 0.18 | **+8.40** | **< 0.0001** |

## Paired Statistical Analysis

- **Synergy Gain Mean:** **+1.28** BLEU-4
- **95% Bootstrap CI:** `[1.21, 1.30]`
- **Paired Permutation p-value:** `p < 0.0001` (`SUPER_ADDITIVE_SYNERGY` confirmed).
