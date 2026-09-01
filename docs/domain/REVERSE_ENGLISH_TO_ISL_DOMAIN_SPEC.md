# Reverse English-to-ISL Domain Specification (ROUTINE_HEALTHCARE_PUBLIC_SERVICE)

## 1. Controlled Intent Scope (8 Core Intents)

- `APPOINTMENT_SCHEDULING`
- `SYMPTOM_REPORT`
- `MEDICATION_INQUIRY`
- `LOCATION_DIRECTION`
- `REGISTRATION_CHECKIN`
- `EMERGENCY_ASSISTANCE`
- `PAYMENT_BILLING`
- `GENERAL_CLARIFICATION`

## 2. Core Entities (12 Entity Types)

- **Doctor**
- **Patient**
- **Nurse**
- **Pharmacist**
- **RegistrationDesk**
- **Pharmacy**
- **Fever**
- **Cough**
- **Pain**
- **Medicine**
- **Morning**
- **Night**

## 3. Parallel Example Dialogues

| ID | Intent | English Sentence | ISL Gloss Sequence | Entities |
| :--- | :--- | :--- | :--- | :--- |
| `diag_001` | `SYMPTOM_REPORT` | "I have a fever today." | **TODAY FEVER HAVE** | Fever |
| `diag_002` | `LOCATION_DIRECTION` | "Where is the pharmacy?" | **PHARMACY WHERE** | Pharmacy |
| `diag_003` | `MEDICATION_INQUIRY` | "Take medicine in the morning." | **MORNING MEDICINE TAKE** | Medicine, Morning |
| `diag_004` | `APPOINTMENT_SCHEDULING` | "I want to book an appointment with doctor." | **DOCTOR APPOINTMENT BOOK WANT** | Doctor |

## 4. Risk Boundary Policy

⚠️ **Risk Boundary:** Any English sentence containing unsupported high-risk terms (e.g., `surgery`, `chemotherapy`, `liability`) is automatically rejected with `REJECTED_RISK_BOUNDARY` status and redirected to human staff.
