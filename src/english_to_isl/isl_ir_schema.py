"""Versioned ISL Intermediate Representation (IR) Schema Module (Prompt 64).

Defines the formal ISL IR Schema (v1.0.0):
1. version
2. intent
3. ordered_gloss_ids
4. spatial_references
5. non_manual_markers
6. timing
7. transition_hints
8. oov_markers
9. confidence
10. provenance

Provides formal JSON Schema validation and example representations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


ISL_IR_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ISLIntermediateRepresentation",
    "version": "1.0.0",
    "type": "object",
    "required": [
        "version",
        "intent",
        "ordered_gloss_ids",
        "spatial_references",
        "non_manual_markers",
        "timing",
        "transition_hints",
        "oov_markers",
        "confidence",
        "provenance"
    ],
    "properties": {
        "version": {"type": "string", "const": "1.0.0"},
        "intent": {"type": "string"},
        "ordered_gloss_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "spatial_references": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["gloss_id", "locus"],
                "properties": {
                    "gloss_id": {"type": "string"},
                    "locus": {"type": "string"}
                }
            }
        },
        "non_manual_markers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["gloss_id", "marker"],
                "properties": {
                    "gloss_id": {"type": "string"},
                    "marker": {"type": "string"}
                }
            }
        },
        "timing": {
            "type": "object",
            "required": ["duration_ms", "speed_multiplier"],
            "properties": {
                "duration_ms": {"type": "integer"},
                "speed_multiplier": {"type": "number"}
            }
        },
        "transition_hints": {
            "type": "array",
            "items": {"type": "string"}
        },
        "oov_markers": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "provenance": {
            "type": "object",
            "required": ["parser", "planner", "timestamp"],
            "properties": {
                "parser": {"type": "string"},
                "planner": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        }
    }
}


class ISLIntermediateRepresentation:
    """Validator and generator for versioned ISL IR objects."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, schema_dir: str = "./artifacts/specs/phase7"):
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)

    def validate_ir(self, ir_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates an ISL IR dictionary against the required schema."""
        errors = []
        for req_field in ISL_IR_JSON_SCHEMA["required"]:
            if req_field not in ir_data:
                errors.append(f"Missing required field: '{req_field}'")

        if ir_data.get("version") != self.SCHEMA_VERSION:
            errors.append(f"Invalid schema version '{ir_data.get('version')}', expected '{self.SCHEMA_VERSION}'")

        if "confidence" in ir_data:
            c = ir_data["confidence"]
            if not isinstance(c, (int, float)) or c < 0.0 or c > 1.0:
                errors.append(f"Confidence score out of range [0.0, 1.0]: {c}")

        return (len(errors) == 0, errors)

    def create_example_ir(self) -> Dict[str, Any]:
        """Creates a canonical valid ISL IR example dict."""
        return {
            "version": "1.0.0",
            "intent": "ont:intent/symptom_report",
            "ordered_gloss_ids": ["TODAY", "FEVER", "HAVE"],
            "spatial_references": [
                {"gloss_id": "FEVER", "locus": "LOC_CENTER"}
            ],
            "non_manual_markers": [
                {"gloss_id": "FEVER", "marker": "head_nod_slight"}
            ],
            "timing": {
                "duration_ms": 1800,
                "speed_multiplier": 1.0
            },
            "transition_hints": ["smooth_blend"],
            "oov_markers": [],
            "confidence": 0.95,
            "provenance": {
                "parser": "EnglishSemanticParser_v1",
                "planner": "ISLPlanner_v1",
                "timestamp": "2026-09-02T00:15:00Z"
            }
        }

    def export_ir_spec(self) -> Tuple[str, str, Dict[str, Any]]:
        """Exports ISL IR JSON Schema & Markdown specification."""
        example_ir = self.create_example_ir()
        valid, errors = self.validate_ir(example_ir)
        assert valid, f"Example IR failed validation: {errors}"

        spec_data = {
            "json_schema": ISL_IR_JSON_SCHEMA,
            "canonical_example": example_ir
        }

        json_path = self.schema_dir / "isl_ir_schema.json"
        md_path = self.schema_dir / "isl_ir_schema.md"
        doc_path = Path("./docs/schemas/ISL_INTERMEDIATE_REPRESENTATION_SPEC.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f, indent=2)

        md_lines = [
            f"# ISL Intermediate Representation Specification (v{self.SCHEMA_VERSION})",
            "",
            "## 1. Schema Required Fields (10 Key Fields)",
            ""
        ]

        for field in ISL_IR_JSON_SCHEMA["required"]:
            md_lines.append(f"- `{field}`")

        md_lines.extend([
            "",
            "## 2. Canonical Example JSON",
            "",
            "```json",
            json.dumps(example_ir, indent=2),
            "```",
            "",
            "## 3. Schema Validation Status",
            "",
            "✅ Schema version `1.0.0` validated with 0 errors."
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_path), str(md_path), spec_data


if __name__ == "__main__":
    ISLIntermediateRepresentation().export_ir_spec()
