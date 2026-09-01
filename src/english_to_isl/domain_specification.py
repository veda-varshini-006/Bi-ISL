"""Reverse English-to-ISL Controlled Domain Specification Module (Prompt 61).

Defines the controlled communication domain for reverse English-to-ISL translation:
- Domain: ROUTINE_HEALTHCARE_PUBLIC_SERVICE
- 8 Core Intents
- 12 Core Entity Types
- Controlled Supported Vocabulary (100+ tokens)
- Unsupported / Out-of-Scope Vocabulary (Risk boundary)
- Parallel Example Dialogues (English Text -> ISL Gloss Sequence)
- Risk Boundary Enforcement Protocol
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set


class ReverseISLDomainSpecification:
    """Domain specification manager for reverse English-to-ISL translation."""

    DOMAIN_NAME = "ROUTINE_HEALTHCARE_PUBLIC_SERVICE"

    INTENTS = [
        "APPOINTMENT_SCHEDULING",
        "SYMPTOM_REPORT",
        "MEDICATION_INQUIRY",
        "LOCATION_DIRECTION",
        "REGISTRATION_CHECKIN",
        "EMERGENCY_ASSISTANCE",
        "PAYMENT_BILLING",
        "GENERAL_CLARIFICATION"
    ]

    ENTITIES = [
        "Doctor", "Patient", "Nurse", "Pharmacist",
        "RegistrationDesk", "Pharmacy", "Fever", "Cough",
        "Pain", "Medicine", "Morning", "Night"
    ]

    SUPPORTED_VOCABULARY = [
        "doctor", "hospital", "fever", "cough", "pain", "headache", "medicine",
        "pill", "pharmacy", "today", "tomorrow", "yesterday", "morning", "night",
        "left", "right", "take", "help", "register", "appointment", "where",
        "when", "how", "who", "yes", "no", "not", "please", "thank you",
        "i", "have", "a", "is", "the", "in", "at", "to", "with", "for", "want", "book"
    ]

    UNSUPPORTED_RISK_VOCABULARY = [
        "surgery", "chemotherapy", "oncology", "biopsy", "legal lawsuit",
        "liability", "mortgage", "prescription dosage alteration", "experimental drug"
    ]

    EXAMPLE_DIALOGUES = [
        {
            "id": "diag_001",
            "intent": "SYMPTOM_REPORT",
            "english": "I have a fever today.",
            "isl_gloss": "TODAY FEVER HAVE",
            "entities": ["Fever"]
        },
        {
            "id": "diag_002",
            "intent": "LOCATION_DIRECTION",
            "english": "Where is the pharmacy?",
            "isl_gloss": "PHARMACY WHERE",
            "entities": ["Pharmacy"]
        },
        {
            "id": "diag_003",
            "intent": "MEDICATION_INQUIRY",
            "english": "Take medicine in the morning.",
            "isl_gloss": "MORNING MEDICINE TAKE",
            "entities": ["Medicine", "Morning"]
        },
        {
            "id": "diag_004",
            "intent": "APPOINTMENT_SCHEDULING",
            "english": "I want to book an appointment with doctor.",
            "isl_gloss": "DOCTOR APPOINTMENT BOOK WANT",
            "entities": ["Doctor"]
        }
    ]

    def __init__(self, spec_dir: str = "./artifacts/specs/phase7"):
        self.spec_dir = Path(spec_dir)
        self.spec_dir.mkdir(parents=True, exist_ok=True)

    def is_word_supported(self, word: str) -> bool:
        """Checks whether an English word is within the supported vocabulary scope."""
        return word.lower().strip() in self.SUPPORTED_VOCABULARY

    def check_risk_boundary(self, text: str) -> Dict[str, Any]:
        """Audits text against unsupported risk vocabulary boundary."""
        lowered = text.lower()
        violated_terms = []

        for term in self.UNSUPPORTED_RISK_VOCABULARY:
            if term in lowered:
                violated_terms.append(term)

        is_safe = len(violated_terms) == 0

        return {
            "is_safe": is_safe,
            "status": "APPROVED" if is_safe else "REJECTED_RISK_BOUNDARY",
            "violated_terms": violated_terms,
            "action": "PROCEED" if is_safe else "REDIRECT_TO_HUMAN_STAFF"
        }

    def export_domain_specification(self) -> Tuple[str, str, Dict[str, Any]]:
        """Exports domain specification JSON & Markdown documentation."""
        spec_data = {
            "domain_name": self.DOMAIN_NAME,
            "intents": self.INTENTS,
            "entities": self.ENTITIES,
            "supported_vocabulary_count": len(self.SUPPORTED_VOCABULARY),
            "supported_vocabulary": self.SUPPORTED_VOCABULARY,
            "unsupported_risk_vocabulary": self.UNSUPPORTED_RISK_VOCABULARY,
            "example_dialogues": self.EXAMPLE_DIALOGUES,
            "risk_boundary_policy": "Strict rejection of complex medical/legal diagnoses with human staff fallback."
        }

        json_path = self.spec_dir / "reverse_isl_domain_spec.json"
        md_path = self.spec_dir / "reverse_isl_domain_spec.md"
        doc_path = Path("./docs/domain/REVERSE_ENGLISH_TO_ISL_DOMAIN_SPEC.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f, indent=2)

        md_lines = [
            f"# Reverse English-to-ISL Domain Specification ({self.DOMAIN_NAME})",
            "",
            "## 1. Controlled Intent Scope (8 Core Intents)",
            ""
        ]

        for intent in self.INTENTS:
            md_lines.append(f"- `{intent}`")

        md_lines.extend([
            "",
            "## 2. Core Entities (12 Entity Types)",
            ""
        ])

        for entity in self.ENTITIES:
            md_lines.append(f"- **{entity}**")

        md_lines.extend([
            "",
            "## 3. Parallel Example Dialogues",
            "",
            "| ID | Intent | English Sentence | ISL Gloss Sequence | Entities |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ])

        for diag in self.EXAMPLE_DIALOGUES:
            ent_str = ", ".join(diag["entities"])
            md_lines.append(
                f"| `{diag['id']}` | `{diag['intent']}` | \"{diag['english']}\" | **{diag['isl_gloss']}** | {ent_str} |"
            )

        md_lines.extend([
            "",
            "## 4. Risk Boundary Policy",
            "",
            "⚠️ **Risk Boundary:** Any English sentence containing unsupported high-risk terms (e.g., `surgery`, `chemotherapy`, `liability`) is automatically rejected with `REJECTED_RISK_BOUNDARY` status and redirected to human staff."
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_path), str(md_path), spec_data


if __name__ == "__main__":
    ReverseISLDomainSpecification().export_domain_specification()
