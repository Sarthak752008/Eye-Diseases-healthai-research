# EyeHealthAI — Privacy Architecture & Clinical Data Governance

**Document Version**: 2.0.0  
**Effective Date**: October 2026  
**Classification**: Investigational Research System • Privacy & Data Protection Specification  

---

## 1. Core Privacy Principles

EyeHealthAI is engineered strictly under a **Privacy-First, Data-Minimization** paradigm compliant with HIPAA, GDPR (Right to Be Forgotten), and clinical research ethics (Declaration of Helsinki):

1. **Zero Permanent Raw Image Storage**:
   - All uploaded photos and live webcam captures are processed strictly in RAM (`io.BytesIO` buffers).
   - Once pre-screening quality checks and ResNet-50 forward inference complete, image buffers are garbage-collected. No ocular images or webcam frames are ever persisted to disk by default.
2. **Explicit Informed Consent Logging**:
   - Every participant session requires explicit opt-in confirmation before screening starts.
   - Timestamped consent records are stored with version tracking (`1.0`) in the `consent_records` database table.
3. **Right to Be Forgotten (GDPR Article 17)**:
   - Participants can permanently purge their profile, session history, and report metadata at any time via `DELETE /api/user/{user_id}`.
   - Foreign key cascading constraints guarantee complete removal of all associated screening and report records.
4. **LLM / Gemini Isolation Firewall**:
   - No patient images, full names, phone numbers, email addresses, or clinical numerical results are ever transmitted to the Gemini API.
   - All prompts are strictly sanitized server-side via regex filters that redact file paths, phone numbers, emails, and patient IDs before LLM dispatch.
   - Diagnostic and prescription queries are blocked deterministically.

---

## 2. Database Schema & Data Retention

The persistent database (SQLite in development, PostgreSQL in production) stores structured non-image records:

| Table | Fields Stored | Retention Policy | Cascades on Deletion |
| :--- | :--- | :--- | :--- |
| `user_profiles` | `id`, `full_name`, `phone`, `email`, `date_of_birth`, `created_at` | Active until participant deletion request | Yes (`CASCADE`) |
| `consent_records` | `id`, `user_id`, `consent_given`, `research_consent`, `data_storage_consent`, `timestamp` | Audit compliance record | Yes (`CASCADE`) |
| `screening_sessions` | `id`, `user_id`, `session_date`, `language`, `selected_module`, `capture_source`, `view_types` | Research session log | Yes (`CASCADE`) |
| `model_results` | `id`, `session_id`, `module`, `operational_status`, `prediction_value`, `unit`, `view_count` | Research metric history | Yes (`CASCADE`) |
| `report_metadata` | `id`, `session_id`, `report_type`, `language`, `generated_at`, `status` | Report tracking | Yes (`CASCADE`) |

**Notice**: Raw image bytes and camera streams are explicitly omitted from the database schema.

---

## 3. Gemini Research Copilot Data Safety Guardrails

The optional Research Copilot adheres to four safety boundaries:

1. **Pre-Transmission Input Sanitization**:
   - Windows/Unix filesystem paths: Redacted to `[PATH_REDACTED]`
   - Hemoglobin values: Redacted to `[CLINICAL_VALUE_REDACTED]`
   - Bilirubin values: Redacted to `[CLINICAL_VALUE_REDACTED]`
   - Phone numbers & email addresses: Redacted to `[PHONE_REDACTED]` / `[EMAIL_REDACTED]`
   - Base64 image payloads: Redacted to `[IMAGE_DATA_REDACTED]`
2. **Deterministic Topic Blocking**:
   - User inputs requesting medical diagnosis, prescription advice, medication dosages, or treatment plans are intercepted locally and returned with a clinical safety notice without querying external LLMs.
3. **Ground Truth Insulation**:
   - The Copilot cannot alter model weights, modify clinical predictions, or bypass reliability gates.
4. **Zero Telemetry**:
   - When `GEMINI_API_KEY` is absent, the system operates completely offline using deterministic documentation retrieval.

---

## 4. Deletion Endpoints & Access Control

- **Delete Participant Account**:
  ```http
  DELETE /api/user/{user_id}
  ```
  Returns HTTP 200 with confirmation that user profile and all linked records have been erased.
- **Delete Single Session**:
  ```http
  DELETE /api/session/{session_id}
  ```
  Removes the specified session and linked model findings.
- **Session IDs**: Unguessable UUIDs (`ses_...`) prevent unauthorized enumeration.
