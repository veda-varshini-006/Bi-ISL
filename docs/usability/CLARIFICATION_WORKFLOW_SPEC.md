# Clarification Workflow & Safety Specification (Prompts 55 & 56)

## Overview & Operational Principles

The **Clarification Engine** (`ClarificationEngine`) works in conjunction with the **Abstention Mechanism** (`AbstentionMechanism`) to replace forced, hallucinated translations with user-friendly, interactive clarification prompts.

---

## 1. Supported Clarification Workflows

- **Visual Uncertainty / Occlusion:**
  - *Trigger:* $p_t < \tau_p$ or Out-of-Vocabulary / Unknown Sign.
  - *Prompt:* `"Please repeat the sign."`
- **Decoding Ambiguity / Dual Top Candidates:**
  - *Trigger:* Top-2 beam decoding candidates have close likelihood probabilities.
  - *Prompt:* `"Did you mean X or Y?"`
- **Low Confidence Fallback:**
  - *Trigger:* High sequence entropy or UGSA adaptation rejection.
  - *Prompt:* `"Translation confidence is low. Please sign again clearly."`

---

## 2. Medical & Legal Safety Guardrails

To prevent generating unsupported medical diagnostic advice or legal liability assertions:
- Clarification prompts strictly prohibit terms such as `diagnose`, `prescribe`, `legal liability`, `guarantee cure`, `treatment plan`, or `verdict`.
- If an unsupported term is detected in high-risk domains (`MEDICAL` / `LEGAL`), the candidate is automatically sanitized to `[UNVERIFIED_TERM]` or replaced with a safe fallback prompt.

---

## 3. Usability Telemetry

- **Usability Metric:** `clarification_frequency` (%)
- **Logging:** All clarification events are appended to `artifacts/logs/usability/clarification_usability.jsonl` with timestamps, domain metadata, and safety audit status.
