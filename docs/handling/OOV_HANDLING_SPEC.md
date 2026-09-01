# Out-of-Vocabulary (OOV) Handling Architecture (Prompt 67)

## Core Safety Constraint: Zero Random Sign Substitution

⚠️ **Strict Rule:**
- The system **NEVER** substitutes a random nearest sign for an unmapped word.
- Random nearest sign substitution in medical or public-service settings creates severe safety risks (e.g. substituting `surgery` for `pill`).
- All unmapped terms are resolved through one of 4 explicit, deterministic resolution modes with full event telemetry logging.

---

## Supported Resolution Modes (4 Modes)

1. **`FINGERSPELLING_VALIDATED`**: Decomposes the word into validated ISL manual alphabet fingerspelling glosses (e.g., `paracetamol` $\rightarrow$ `["FS_P", "FS_A", "FS_R", "FS_A", "FS_C", "FS_E", "FS_T", "FS_A", "FS_M", "FS_O", "FS_L"]`).
2. **`CLARIFICATION_REQUEST`**: Triggers a non-forced dialogue clarification prompt requesting user rephrasing.
3. **`TEXT_DISPLAY_FALLBACK`**: Renders an explicit text caption overlay for the unsigned term alongside avatar motion.
4. **`EXPLICIT_OOV_STATE`**: Enters an explicit `OOV_HALT` state to halt motion generation safely with zero hallucination.

---

## Telemetry Log Location

All OOV events are appended as structured JSONL records to:
- [`artifacts/logs/oov/oov_events.jsonl`](file:///c:/Users/ADMIN/Downloads/ARVR/artifacts/logs/oov/oov_events.jsonl)
