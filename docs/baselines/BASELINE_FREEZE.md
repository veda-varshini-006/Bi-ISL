# Bi-ISL Official Baseline Freeze Document (BASELINE_V1)

> [!IMPORTANT]
> **Baseline Rule:** No proposed Bi-ISL mechanism (SBDS or UGSA) may be evaluated against a deliberately weaker baseline. All future experimental mechanisms MUST be evaluated against `BASELINE_V1`.

## Frozen Specifications (`BASELINE_V1`)

- **Git Tag:** `BASELINE_V1`
- **Commit Hash:** `cfb2504c54c90ed0d3f557281f4832f24de1ca03`
- **Selected Model:** `MultimodalBaseline (RGB + Hands + Pose + Face)`
- **Validation BLEU-4:** **40.0 ± 0.0** (Evaluated over seeds `[42, 123, 456]` using validation data only)

### Frozen Component Registry

| Component | Frozen Path / Reference | Version / Hash |
| :--- | :--- | :--- |
| **Configuration** | `config/base_config.yaml` | SHA-256 Verified |
| **Data Split** | `artifacts/splits/split_manifest.json` | Immutable Manifest |
| **Tokenizer** | `src/text/tokenizer.py` | `v1.0.0_word_level` |
| **Evaluation Suite** | `src/evaluation/e1_evaluator.py` | E1 Multi-Metric |
| **Model Checkpoint** | `artifacts/checkpoints/baseline_v1/best_checkpoint.pt` | Saved State Dict |
| **Commit Hash** | `cfb2504c54c90ed0d3f557281f4832f24de1ca03` | Git HEAD SHA |

## Multi-Seed Validation Results

| Seed | BLEU-4 | chrF++ | Semantic Score (v1) | Mean WER |
| :---: | :---: | :---: | :---: | :---: |
| `42` | 40.0 | 97.33 | 92.0 | 5.0% |
| `123` | 40.0 | 97.33 | 92.0 | 5.0% |
| `456` | 40.0 | 97.33 | 92.0 | 5.0% |

✅ **BASELINE_V1 officially frozen and tagged.**
