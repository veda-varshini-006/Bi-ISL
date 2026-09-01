# Formal Domain Ontology Specification (Prompt 62)

## 1. Stable Language-Independent Concept URIs

### Intents (`ont:intent/*`)

- `ont:intent/symptom_report`: Report physical symptom or condition
- `ont:intent/location_inquiry`: Ask for location or direction
- `ont:intent/medication_instruction`: Instruct medication intake timing
- `ont:intent/appointment_booking`: Schedule consultation appointment
- `ont:intent/emergency_request`: Request immediate emergency assistance

### Entities (`ont:entity/*`)

- `ont:entity/medical_practitioner`: Healthcare professional (Doctor/Nurse)
- `ont:entity/patient`: Person receiving healthcare
- `ont:entity/medication`: Pharmaceutical drug or treatment
- `ont:entity/symptom_fever`: Elevated body temperature symptom
- `ont:entity/symptom_cough`: Respiratory cough symptom
- `ont:entity/symptom_pain`: Physical pain symptom

### Relationships (`ont:rel/*`)

| Subject | Relationship | Object |
| :--- | :--- | :--- |
| `ont:entity/patient` | `ont:rel/experiences` | `ont:entity/symptom_fever` |
| `ont:entity/medical_practitioner` | `ont:rel/prescribes` | `ont:entity/medication` |
| `ont:entity/medication` | `ont:rel/located_at` | `ont:loc/pharmacy` |

## 2. Decoupling Guarantee

✅ All concepts use stable `ont:*` URIs independent of surface English vocabulary and avatar 3D keypoint clip representations.
