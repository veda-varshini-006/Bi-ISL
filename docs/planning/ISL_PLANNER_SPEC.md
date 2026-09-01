# ISL Planner Architecture Specification (Prompt 65)

## 4-Phase Transformation Pipeline

The **ISL Planner** (`ISLPlanner`) converts a JSON Semantic Frame into a schema-validated **ISL Intermediate Representation (IR)** through 4 explicit, decoupled phases:

```
[JSON Semantic Frame]
       │
       ▼
Phase 1: Semantic Transformation
       │
       ▼
Phase 2: Word/Sign Ordering (ISL Grammar Rules: Time -> Topic -> Comment / SOV)
       │
       ▼
Phase 3: Non-Manual Markers (Facial Expressions & Eyebrow Annotations)
       │
       ▼
Phase 4: Avatar Motion Hints (Spatial Loci & Co-articulation Transitions)
       │
       ▼
[ISL IR Object (v1.0.0)]
```

---

## Supported Intent Mappings (8 Intents)

| Intent URI | English Input | Generated ISL Gloss Sequence | Non-Manual Marker |
| :--- | :--- | :--- | :--- |
| `ont:intent/symptom_report` | "I have a fever today." | `TODAY FEVER HAVE` | `head_nod_slight` |
| `ont:intent/location_inquiry` | "Where is the pharmacy?" | `PHARMACY WHERE` | `eyebrows_furrowed` |
| `ont:intent/medication_instruction` | "Take medicine in the morning." | `MORNING MEDICINE TAKE` | `neutral_facial` |
| `ont:intent/appointment_booking` | "Book appointment with doctor." | `DOCTOR APPOINTMENT BOOK WANT` | `head_nod_slight` |
| `ont:intent/emergency_request` | "I need urgent help!" | `HELP EMERGENCY URGENT` | `eyes_wide_expressive` |
| `ont:intent/registration_checkin` | "I want to register." | `PATIENT REGISTER WANT` | `neutral_facial` |
| `ont:intent/payment_billing` | "Where do I pay bill?" | `BILL PAYMENT WHERE` | `eyebrows_furrowed` |
| `ont:intent/general_clarification` | "Please repeat." | `PLEASE REPEAT` | `eyebrows_raised` |
