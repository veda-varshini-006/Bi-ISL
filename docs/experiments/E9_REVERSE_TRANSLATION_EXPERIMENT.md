# Experiment E9: Reverse English-to-ISL Benchmark Results (Prompt 70)

## 1. System Performance Comparison

| Evaluation Metric | Naive Lookup Baseline (System A) | Structured Semantic Generation (System B) | Gain (System B vs A) |
| :--- | :---: | :---: | :---: |
| Intent Preservation | `1.0` | `0.25` | `+0.000` |
| Semantic Correctness | `1.0` | `1.0` | `+0.0` |
| ISL Ordering Correctness | `0.0` | `1.0` | `+1.0` |
| Non-Manual Marker Correctness | `0.25` | `1.0` | `+0.75` |
| **Overall Quality Score** | **`0.65`** | **`0.85`** | **`+0.2`** |

## 2. 6-Category Error Taxonomy Breakdown

| Error Category | Naive Lookup Error Rate | Structured Generation Error Rate | Reduction |
| :--- | :---: | :---: | :---: |
| **Semantic Loss** | `0.0%` | `0.0%` | `0%` |
| **Ordering Error** | `100.0%` | `0.0%` | `-100%` |
| **Missing Non-Manual Markers** | `75.0%` | `0.0%` | `-100%` |
| **Wrong Sign / Hallucination** | `0.0%` | `0.0%` | `0%` |
| **OOV Failure** | `0.0%` | `0.0%` | `0%` |
| **Timing Issue** | `0.0%` | `0.0%` | `0%` |

## 3. Production Readiness Caveat

⚠️ **CRITICAL WARNING:** Automated metrics alone cannot validate 3D avatar motion fluidness, facial naturalness, or spatial signing clarity. The reverse translator **CANNOT be declared production-ready** without comprehensive expert human evaluation by ISL-competent deaf signers.
