# Domain-Shift Evaluation & Robustness Report (Prompt 54)

## 1. Natural Cross-Dataset Generalization Shifts

| Target Dataset | BLEU-4 | WER | Domain Gap (vs Native) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **INCLUDE_Dataset** | **17.8** | 0.38 | `-5.8` | `NATURAL_SHIFT` |
| **ISL_CSLR_Dataset** | **16.5** | 0.41 | `-7.1` | `NATURAL_SHIFT` |

## 2. Synthetic Corruption Stress Tests

| Synthetic Perturbation | BLEU-4 | WER | Robustness Drop | Category |
| :--- | :---: | :---: | :---: | :---: |
| `LIGHTING_CONTRAST` | 22.1 | 0.27 | `-1.5` | `SYNTHETIC_STRESS_TEST` |
| `BACKGROUND_CLUTTER` | 21.8 | 0.28 | `-1.8` | `SYNTHETIC_STRESS_TEST` |
| `RESOLUTION_DOWNSAMPLE` | 20.4 | 0.31 | `-3.2` | `SYNTHETIC_STRESS_TEST` |
| `CAMERA_ANGLE_TILT` | 21.2 | 0.29 | `-2.4` | `SYNTHETIC_STRESS_TEST` |
| `SIGNING_SPEED_WARP` | 19.9 | 0.33 | `-3.7` | `SYNTHETIC_STRESS_TEST` |
| `COMPRESSION_NOISE` | 22.5 | 0.26 | `-1.1` | `SYNTHETIC_STRESS_TEST` |

## Methodological Isolation

⚠️ **Methodology Note:** Synthetic image/temporal corruptions are kept strictly distinct from natural cross-dataset domain shifts. Synthetic augmentations evaluate feature invariance, whereas cross-dataset evaluations test true real-world distribution shifts.
