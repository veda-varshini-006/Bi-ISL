"""Formal Domain Ontology Module (Prompt 62).

Represents healthcare & public-service domain concepts via stable, language-independent identifiers:
- Intents (ont:intent/...)
- Entities (ont:entity/...)
- Slots (ont:slot/...)
- Relationships (ont:rel/...)
- Question Types (ont:qtype/...)
- Negation (ont:negation/...)
- Temporality (ont:temp/...)
- Locations (ont:loc/...)
- Actions (ont:action/...)

Independent of English surface wording and avatar clips.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set


class DomainOntology:
    """Formal domain ontology manager."""

    ONTOLOGY_URI_PREFIX = "ont:"

    INTENTS = {
        "ont:intent/symptom_report": "Report physical symptom or condition",
        "ont:intent/location_inquiry": "Ask for location or direction",
        "ont:intent/medication_instruction": "Instruct medication intake timing",
        "ont:intent/appointment_booking": "Schedule consultation appointment",
        "ont:intent/emergency_request": "Request immediate emergency assistance"
    }

    ENTITIES = {
        "ont:entity/medical_practitioner": "Healthcare professional (Doctor/Nurse)",
        "ont:entity/patient": "Person receiving healthcare",
        "ont:entity/medication": "Pharmaceutical drug or treatment",
        "ont:entity/symptom_fever": "Elevated body temperature symptom",
        "ont:entity/symptom_cough": "Respiratory cough symptom",
        "ont:entity/symptom_pain": "Physical pain symptom"
    }

    SLOTS = {
        "ont:slot/target_entity": "Primary entity target",
        "ont:slot/location_target": "Spatial destination or area",
        "ont:slot/temporal_anchor": "Time frame or time-of-day",
        "ont:slot/severity_level": "Degree or intensity of symptom"
    }

    RELATIONSHIPS = [
        {"subject": "ont:entity/patient", "relation": "ont:rel/experiences", "object": "ont:entity/symptom_fever"},
        {"subject": "ont:entity/medical_practitioner", "relation": "ont:rel/prescribes", "object": "ont:entity/medication"},
        {"subject": "ont:entity/medication", "relation": "ont:rel/located_at", "object": "ont:loc/pharmacy"}
    ]

    QUESTION_TYPES = {
        "ont:qtype/location_where": "Where question seeking spatial location",
        "ont:qtype/time_when": "When question seeking temporal anchor",
        "ont:qtype/identity_who": "Who question seeking entity identity",
        "ont:qtype/binary_yesno": "Yes/No confirmation question"
    }

    NEGATION_TYPES = {
        "ont:negation/absent": "Indicates absence of symptom or entity",
        "ont:negation/prohibited": "Indicates prohibition or restriction"
    }

    TEMPORALITY_TYPES = {
        "ont:temp/past_yesterday": "Past time anchor (yesterday)",
        "ont:temp/present_now": "Present time anchor (now/today)",
        "ont:temp/future_tomorrow": "Future time anchor (tomorrow)",
        "ont:temp/tod_morning": "Time of day (morning)",
        "ont:temp/tod_night": "Time of day (night)"
    }

    LOCATIONS = {
        "ont:loc/pharmacy": "Dispensary or pharmacy area",
        "ont:loc/hospital_reception": "Hospital reception or check-in desk",
        "ont:loc/clinic": "Outpatient clinic or doctor's office"
    }

    ACTIONS = {
        "ont:action/take_medication": "Ingest or apply prescribed medication",
        "ont:action/book_appointment": "Reserve a time slot with practitioner",
        "ont:action/seek_help": "Request urgent staff intervention"
    }

    def __init__(self, ontology_dir: str = "./artifacts/specs/phase7"):
        self.ontology_dir = Path(ontology_dir)
        self.ontology_dir.mkdir(parents=True, exist_ok=True)

    def resolve_concept(self, uri: str) -> Optional[str]:
        """Resolves stable ontology URI identifier to concept definition."""
        if uri.startswith("ont:intent/"):
            return self.INTENTS.get(uri)
        if uri.startswith("ont:entity/"):
            return self.ENTITIES.get(uri)
        if uri.startswith("ont:slot/"):
            return self.SLOTS.get(uri)
        if uri.startswith("ont:qtype/"):
            return self.QUESTION_TYPES.get(uri)
        if uri.startswith("ont:negation/"):
            return self.NEGATION_TYPES.get(uri)
        if uri.startswith("ont:temp/"):
            return self.TEMPORALITY_TYPES.get(uri)
        if uri.startswith("ont:loc/"):
            return self.LOCATIONS.get(uri)
        if uri.startswith("ont:action/"):
            return self.ACTIONS.get(uri)
        return None

    def export_ontology(self) -> Tuple[str, str, Dict[str, Any]]:
        """Exports formal domain ontology JSON & Markdown documentation."""
        ontology_data = {
            "ontology_uri_prefix": self.ONTOLOGY_URI_PREFIX,
            "intents": self.INTENTS,
            "entities": self.ENTITIES,
            "slots": self.SLOTS,
            "relationships": self.RELATIONSHIPS,
            "question_types": self.QUESTION_TYPES,
            "negation_types": self.NEGATION_TYPES,
            "temporality_types": self.TEMPORALITY_TYPES,
            "locations": self.LOCATIONS,
            "actions": self.ACTIONS
        }

        json_path = self.ontology_dir / "domain_ontology.json"
        md_path = self.ontology_dir / "domain_ontology.md"
        doc_path = Path("./docs/ontology/DOMAIN_ONTOLOGY_SPEC.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ontology_data, f, indent=2)

        md_lines = [
            "# Formal Domain Ontology Specification (Prompt 62)",
            "",
            "## 1. Stable Language-Independent Concept URIs",
            "",
            "### Intents (`ont:intent/*`)",
            ""
        ]

        for uri, desc in self.INTENTS.items():
            md_lines.append(f"- `{uri}`: {desc}")

        md_lines.extend([
            "",
            "### Entities (`ont:entity/*`)",
            ""
        ])

        for uri, desc in self.ENTITIES.items():
            md_lines.append(f"- `{uri}`: {desc}")

        md_lines.extend([
            "",
            "### Relationships (`ont:rel/*`)",
            "",
            "| Subject | Relationship | Object |",
            "| :--- | :--- | :--- |"
        ])

        for rel in self.RELATIONSHIPS:
            md_lines.append(f"| `{rel['subject']}` | `{rel['relation']}` | `{rel['object']}` |")

        md_lines.extend([
            "",
            "## 2. Decoupling Guarantee",
            "",
            "✅ All concepts use stable `ont:*` URIs independent of surface English vocabulary and avatar 3D keypoint clip representations."
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return str(json_path), str(md_path), ontology_data


if __name__ == "__main__":
    DomainOntology().export_ontology()
