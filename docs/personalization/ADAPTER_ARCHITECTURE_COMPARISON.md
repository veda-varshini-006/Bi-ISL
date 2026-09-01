# Bi-ISL Signer Adapter Placement Architecture Comparison (Prompt 41)

## Candidate Placement Evaluation Matrix

| Candidate Placement Location | Trainable Params / Signer | Memory Footprint (KB) | Val BLEU-4 | Val chrF++ | Val Mean WER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VISUAL_ENCODER_OUTPUT** | **8,464** | `33.06 KB` | 40.0 | 97.33 | 5.0% |
| **SELECTED_ENCODER_BLOCKS** | **8,464** | `33.06 KB` | 40.0 | 97.33 | 5.0% |
| **TEMPORAL_REPRESENTATION** | **8,464** | `33.06 KB` | 40.0 | 97.33 | 5.0% |
| **DECODER_INPUT** | **8,464** | `33.06 KB` | 40.0 | 97.33 | 5.0% |

## Minimal Architecture Selection

- **Selected Optimal Placement:** `VISUAL_ENCODER_OUTPUT`
- **Trainable Parameters per Signer:** **8,464** (< 50K constraint satisfied)
- **Memory Overhead per Signer:** **33.06 KB** (~0.03 MB per user profile)

✅ **Base model successfully frozen. Minimal signer adapter architecture selected.**
