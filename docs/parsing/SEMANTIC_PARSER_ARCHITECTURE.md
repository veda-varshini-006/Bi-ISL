# English Semantic Parser Architecture & Guardrails (Prompt 63)

## Architectural Synthesis

The **English Semantic Parser** (`EnglishSemanticParser`) converts free-form English input text into structured **JSON Semantic Frames** with explicit confidence estimation and risk-boundary validation.

---

## Strict Guardrail: No Direct Word-to-Clip Mapping

⚠️ **Core Architectural Constraint:**
- The parser **NEVER** maps directly from arbitrary English surface words to 3D avatar motion video clips.
- Direct surface mapping leads to ungrammatical ISL ordering, missing non-manual facial markers, and hallucinated clips.
- All translation flows **MUST** pass through a validated JSON Semantic Frame adhering to the domain ontology (`ont:*` URIs).

---

## JSON Semantic Frame Schema Example

```json
{
  "english_text": "Where is the pharmacy?",
  "intent_uri": "ont:intent/location_inquiry",
  "entities": [
    {
      "entity_uri": "ont:loc/pharmacy",
      "slot": "location_target",
      "surface_word": "pharmacy"
    }
  ],
  "question_type_uri": "ont:qtype/location_where",
  "negation_type_uri": null,
  "temporality_uri": null,
  "confidence_score": 1.0,
  "parse_status": "SUCCESS",
  "direct_clip_mapping_allowed": false
}
```
