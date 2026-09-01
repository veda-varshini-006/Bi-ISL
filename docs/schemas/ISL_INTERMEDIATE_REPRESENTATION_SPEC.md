# ISL Intermediate Representation Specification (v1.0.0)

## 1. Schema Required Fields (10 Key Fields)

- `version`
- `intent`
- `ordered_gloss_ids`
- `spatial_references`
- `non_manual_markers`
- `timing`
- `transition_hints`
- `oov_markers`
- `confidence`
- `provenance`

## 2. Canonical Example JSON

```json
{
  "version": "1.0.0",
  "intent": "ont:intent/symptom_report",
  "ordered_gloss_ids": [
    "TODAY",
    "FEVER",
    "HAVE"
  ],
  "spatial_references": [
    {
      "gloss_id": "FEVER",
      "locus": "LOC_CENTER"
    }
  ],
  "non_manual_markers": [
    {
      "gloss_id": "FEVER",
      "marker": "head_nod_slight"
    }
  ],
  "timing": {
    "duration_ms": 1800,
    "speed_multiplier": 1.0
  },
  "transition_hints": [
    "smooth_blend"
  ],
  "oov_markers": [],
  "confidence": 0.95,
  "provenance": {
    "parser": "EnglishSemanticParser_v1",
    "planner": "ISLPlanner_v1",
    "timestamp": "2026-09-02T00:15:00Z"
  }
}
```

## 3. Schema Validation Status

✅ Schema version `1.0.0` validated with 0 errors.
