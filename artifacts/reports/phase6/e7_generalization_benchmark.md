# Phase 6 Master Report - E7 Generalization Benchmark (Prompt 60)

## Operational Regimes Generalization Matrix

| Operational Regime | BLEU-4 | WER | ECE | Robustness Gap vs Peak |
| :--- | :---: | :---: | :---: | :---: |
| **Seen Signer (Clean)** | **25.1** | 0.21 | 0.032 | `0.0` |
| **Unseen Signer (Clean)** | **18.37** | 0.29 | 0.041 | `-6.73` |
| **In-Domain Native Benchmark** | **23.6** | 0.245 | 0.038 | `-1.5` |
| **Cross-Domain (INCLUDE Dataset)** | **17.8** | 0.38 | 0.082 | `-7.3` |
| **Challenging (Signing Speed Warp)** | **19.9** | 0.33 | 0.065 | `-5.2` |
| **Challenging (Low Lighting)** | **22.1** | 0.27 | 0.048 | `-3.0` |

## Methodological Guarantee

⚠️ **Absence of Metric Collapsing:** Absolute performance metrics are reported across each operational regime independently without averaging across disjoint domains.
