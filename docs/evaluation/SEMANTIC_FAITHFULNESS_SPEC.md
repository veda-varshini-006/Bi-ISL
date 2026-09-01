# Semantic Faithfulness Evaluation Report (Prompt 57)

## Summary Metrics Grid (BLEU / chrF / Slot F1)

| Metric | Score | Category |
| :--- | :---: | :--- |
| **BLEU-4** | `23.6` | Surface N-Gram Overlap |
| **chrF** | `54.2` | Character N-Gram F-Score |
| **Semantic Slot F1** | **`0.834`** | Meaning Preservation |
| **Semantic Precision** | `0.834` | Hallucination Control |
| **Semantic Recall** | `0.834` | Information Retention |

## Per-Slot Category Preservation Matrix

| Slot Category | Slot F1 Score | Description |
| :--- | :---: | :--- |
| `intent` | **1.0** | Intent / Domain Slot |
| `entity` | **1.0** | Intent / Domain Slot |
| `location` | **1.0** | Intent / Domain Slot |
| `symptom_object` | **1.0** | Intent / Domain Slot |
| `direction` | **1.0** | Intent / Domain Slot |
| `time` | **1.0** | Intent / Domain Slot |
| `negation` | **1.0** | Intent / Domain Slot |
| `question_type` | **1.0** | Intent / Domain Slot |
