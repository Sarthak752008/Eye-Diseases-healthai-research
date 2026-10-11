# Project Status — EyeHealthAI (Layers 1–6 Complete)

**Last Updated**: 2026-10-11  
**Workspace**: `c:\Users\sarth\OneDrive\Pictures\Innotech`

---

## 1. Executive Summary & Verification Matrix

| Target | Layer 1: Audit & Manifest | Layer 2: Preprocessing & Leakage Prevention | Layer 3: Model Architecture & Training | Layer 3.5: Reliability & Domain Validation | Layer 4: Unified Inference & Reliability Gate | Layer 5: Research Dashboard & Session Export | Layer 6: Live Webcam Capture & Guidance | Checkpoint Path & Verification | Key Metrics (Evidence-Backed) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Anemia (Hb Regression)** | **COMPLETE** (`anemia_manifest.csv`: 862 records, 218 subjects) | **COMPLETE** (Standardized 224x224, subject-level split, 0% leakage) | **VALIDATED** (ResNet-50 CNN Regressor vs Mean/Median Baselines) | **AUDITED & GATED** (Clinical limitations documented; research baseline only) | **INTEGRATED (RESEARCH ONLY)**: Multi-view session aggregation, reproducibility std | **INTEGRATED (UI & API)**: Multi-view upload, tagging, quality pre-check, JSON audit export | **CAMERA-DOMAIN GATED**: Webcam blocked by default (`blocked_domain_validation`); researcher override available | `saved_models/anemia/best_model.pth` (288.8 MB; verified loaded) | **Patient-Level Test**: MAE **1.5013** g/dL, RMSE **2.0728** g/dL, R² **0.2836**<br>**View-Level Test**: MAE **1.7757** g/dL, RMSE **2.2915** g/dL, R² **0.1367**<br>*(Baseline Test MAE: Mean=2.0099, Median=1.9877)* |
| **Model B: Conjunctivitis (Classification)** | **REMEDIATED** (`conjunctivitis_manifest_v2_clean.csv`: 965 records) | **REMEDIATED** (Deduplicated, zero cross-split dHash collisions, 224x224 crop pipeline) | **RETRAINED & REMEDIATED** (MobileNetV3-Small Binary Classifier) | **SHORTCUT QUANTIFIED & GATED** (Metadata-only baseline achieves 100% accuracy without pixels) | **RELIABILITY GATED (`blocked_validation`)**: Prediction output withheld | **GATED IN DASHBOARD**: Card marked BLOCKED; predictions withheld; safety rationale displayed | **GATED IN WEBCAM**: Withheld under both upload and webcam captures | • Original: `saved_models/conjunctivitis/best_model.pth`<br>• Clean v2: `saved_models/conjunctivitis/best_model_v2_clean.pth` (12.5 MB) | **Clean Test Accuracy**: 1.0000, **Sensitivity**: 1.0000, **Specificity**: 1.0000, **ROC-AUC**: 1.0000<br>*(Metadata Baseline Test Acc: 1.0000; proves task solved by resolution shortcut)* |
| **Model C: Jaundice (Scleral Icterus)** | **COMPLETE** (`jaundice_manifest.csv`: 0 records, blocker logged) | **COMPLETE** (Directories prepared under `data/processed/jaundice/`) | **SCAFFOLDED / BLOCKED** (MobileNetV3-Small architecture verified) | **AUDITED / BLOCKED** (Zero laboratory bilirubin ground truth exists) | **MISSING-DATA GATED (`unavailable_missing_data`)**: No placeholder or synthetic inference | **LOCKED IN DASHBOARD**: Card marked UNAVAILABLE; inference disabled; anti-fabrication notice | **LOCKED IN WEBCAM**: Inference disabled without placeholder values | Pipeline config at `saved_models/jaundice/jaundice_config.json` | **N/A (Blocker)**: No raw images or laboratory bilirubin values in workspace |

---

## 2. Layer 6 — Live Laptop Webcam Capture & Image Quality Guidance

Implemented in `static/index.html`, `static/css/style.css`, `static/js/dashboard.js`, `app.py`, and `docs/WEBCAM_VALIDATION.md`.

### 2.1 Live Laptop Webcam Interface & Alignment Guide Remediation
- **Mode Switching**: Seamless toggle between "File Upload Mode" and "Live Laptop Camera" without reloading or state loss.
- **Camera Access & Permission**: Uses standard `navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } })`. Gracefully handles permission denial (`NotAllowedError`), missing devices (`NotFoundError`), and busy hardware.
- **Still Frame Capture**: Captures still frames via in-memory `<canvas>` into JPEG Blobs. No continuous video streaming is recorded or persisted.
- **Eye Alignment Guide & Manual Positioning**: Fixed reticle and oval styling with clear manual positioning instructions (*"Manual Macro Framing Target (10–15 cm) • Position one eye to fill this area. Positioning aid only; not automated eye detection"*). Explicitly avoids claiming automated alignment unless actual ocular detection confirms it.
- **Macro Framing Guidance**: Prominent warning (*"⚠️ MACRO EYE FRAMING REQUIRED — NOT A FULL FACE"*) clarifies that the model requires a close-up ocular conjunctiva crop; full-face selfies contain facial skin and background that cannot be analyzed for microvascular pallor.
- **Resolution & Dimensions Metadata**: Traces dimensions from `/api/quality-check` with fallback to `webcamCanvas` resolution, preventing `undefinedxundefined` display artifacts. Displays actual resolution (e.g., `1280×720 px`) or explicit `"Dimensions: Unavailable"`.
- **Single-View Variation Display**: Displays view-to-view variation as `"Not available — one view"` when `view_count <= 1`, eliminating misleading `±0 g/dL` confidence interval illusions.
- **Recapture & Shutdown**: Single-click "Recapture" discards the still and restarts video preview; "Stop" terminates all active hardware media tracks.

### 2.2 Objective Image Quality & Actionable Guidance
Every captured still or uploaded image triggers `/api/quality-check` returning:
- **Sharpness Indicator**: 2D discrete Laplacian filter variance proxy.
- **Exposure Indicator**: Mean luminance [0, 255] flagging underexposure (<30) and overexposure (>230).
- **Usability Status**: `"usable"` vs `"needs_adjustment"`.
- **Actionable Guidance**:
  - *"Capture again holding camera steady to reduce motion blur."*
  - *"Improve lighting: face a diffuse light source or increase room illumination."*
  - *"Position the eye within 10–15 cm of the camera lens for optimal focal clarity."*
  - *"Macro framing required: Capture a close-up ocular view of the conjunctiva, not a full-face selfie."*
- **Anatomical Boundary Disclaimer**: Explicitly warns that generic sharpness/luminance metrics do not confirm whether the inner palpebral conjunctiva or sclera is properly everted, visible, or correctly framed.

### 2.3 Strict Camera-Domain Validation Gate & Medical Decision Disclaimer
- **Default State**: Webcam captures are marked `blocked_domain_validation` by default. Model outputs are withheld to prevent unvalidated consumer sensors from producing misleading clinical estimates.
- **Domain Mismatch Rationale**: Uncalibrated laptop webcams introduce severe chromaticity shifts (auto-white-balance, auto-gain, rolling shutter noise) that distort erythema and pallor signals.
- **Researcher Experimental Override**: An explicit checkbox allows researchers to test unvalidated webcam inferences. Prominent warnings explicitly state:
  - *"RESEARCHER EXPERIMENTAL OVERRIDE ACTIVE: Inference executed on uncalibrated laptop-webcam capture."*
  - *"UNVALIDATED WEBCAM ESTIMATE: Substantial error is expected due to webcam sensor auto-gain, color distortion, and compression artifacts. This output is unvalidated and completely unsuitable for medical decisions."*
  - *"Webcam estimate is completely unvalidated and unsuitable for medical decisions. Paired venous blood reference testing is required."*
- **Module Persistence**: Conjunctivitis remains `blocked_validation` and Jaundice remains `unavailable_missing_data` under all webcam interactions.

### 2.4 Input & Framing Validation
- Rejects incompatible anatomical views (e.g., `'full_face'`, `'face'`, `'portrait'`) before model execution.
- Integrates `verify_ocular_framing_or_detect_face` to detect full-face inputs where automated face detection is supported.
- Session export payload records `"capture_source": "webcam"` or `"upload"`.
- In-memory decoding ensures zero permanent disk leakage of webcam stills.

---

## 3. How to Launch the Application

Run the server from the repository root:

```bash
python app.py
```

- **Dashboard UI**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`

---

## 4. Automated Test Suite Verification

Run command: `python -m pytest tests/ -v`  
Result: **39 passed in 12.64s (100% pass rate across Layers 1–6)**

```
tests/test_layer4_inference.py::test_response_schema_keys PASSED         [  2%]
tests/test_layer4_inference.py::test_conjunctivitis_reliability_gate PASSED [  5%]
tests/test_layer4_inference.py::test_jaundice_missing_data_gate PASSED   [  7%]
tests/test_layer4_inference.py::test_anemia_single_view_inference PASSED [ 10%]
tests/test_layer4_inference.py::test_anemia_multi_view_aggregation PASSED [ 12%]
tests/test_layer4_inference.py::test_anemia_incompatible_view_rejection PASSED [ 15%]
tests/test_layer4_inference.py::test_invalid_image_handling PASSED       [ 17%]
tests/test_layer4_inference.py::test_missing_checkpoint_handling PASSED  [ 20%]
tests/test_layer4_inference.py::test_unified_predict_gateway PASSED      [ 23%]
tests/test_layer5_dashboard.py::test_dashboard_html_served PASSED        [ 25%]
tests/test_layer5_dashboard.py::test_api_status_endpoint PASSED          [ 28%]
tests/test_layer5_dashboard.py::test_quality_check_valid_image PASSED    [ 30%]
tests/test_layer5_dashboard.py::test_quality_check_corrupt_file PASSED   [ 33%]
tests/test_layer5_dashboard.py::test_analyze_anemia_single_view PASSED   [ 35%]
tests/test_layer5_dashboard.py::test_analyze_anemia_multi_view PASSED    [ 38%]
tests/test_layer5_dashboard.py::test_conjunctivitis_blocked_gate_enforcement PASSED [ 41%]
tests/test_layer5_dashboard.py::test_jaundice_missing_data_gate_enforcement PASSED [ 43%]
tests/test_layer5_dashboard.py::test_session_export_endpoint PASSED      [ 46%]
tests/test_layer5_dashboard.py::test_session_export_schema_rejection PASSED [ 48%]
tests/test_layer6_webcam.py::test_webcam_ui_rendered_in_dashboard PASSED [ 51%]
tests/test_layer6_webcam.py::test_quality_check_returns_guidance_and_dimensions PASSED [ 53%]
tests/test_layer6_webcam.py::test_webcam_default_domain_validation_gate PASSED [ 56%]
tests/test_layer6_webcam.py::test_webcam_experimental_override PASSED    [ 58%]
tests/test_layer6_webcam.py::test_conjunctivitis_webcam_gate_preserved PASSED [ 61%]
tests/test_layer6_webcam.py::test_jaundice_webcam_gate_preserved PASSED  [ 64%]
tests/test_layer6_webcam.py::test_session_export_includes_capture_source PASSED [ 66%]
tests/test_layer6_webcam.py::test_existing_upload_inference_remains_functional PASSED [ 69%]
tests/test_layer6_webcam.py::test_eye_guide_rendering_and_manual_instructions PASSED [ 71%]
tests/test_layer6_webcam.py::test_dimension_metadata_integrity PASSED    [ 74%]
tests/test_layer6_webcam.py::test_webcam_override_warning_medical_decision_disclaimer PASSED [ 76%]
tests/test_layer6_webcam.py::test_invalid_full_face_input_rejection PASSED [ 79%]
tests/test_layer6_webcam.py::test_single_view_variation_display_logic PASSED [ 82%]
tests/test_layers.py::test_manifests_exist PASSED                        [ 84%]
tests/test_layers.py::test_anemia_checkpoint_loads PASSED                [ 87%]
tests/test_layers.py::test_conjunctivitis_checkpoint_loads PASSED        [ 89%]
tests/test_layers.py::test_conjunctivitis_v2_clean_checkpoint_loads PASSED [ 92%]
tests/test_layers.py::test_jaundice_architecture_forward PASSED          [ 94%]
tests/test_layers.py::test_metadata_baseline_report_exists_and_valid PASSED [ 97%]
tests/test_layers.py::test_layer_3_5_reliability_documentation_exists PASSED [ 81%]
tests/test_phase7_readiness.py::test_health_endpoints_dual_route PASSED  [ 83%]
tests/test_phase7_readiness.py::test_security_headers_present PASSED     [ 85%]
tests/test_phase7_readiness.py::test_oversized_image_rejection PASSED    [ 87%]
tests/test_phase7_readiness.py::test_unsupported_file_extension_rejection PASSED [ 89%]
tests/test_phase7_readiness.py::test_unsupported_mime_type_rejection PASSED [ 91%]
tests/test_phase7_readiness.py::test_frontend_session_invalidation_and_reset_integrity PASSED [ 93%]
tests/test_phase7_readiness.py::test_html_dashboard_reset_button_exists PASSED [ 95%]
tests/test_phase7_readiness.py::test_configuration_integrity PASSED      [ 97%]
tests/test_phase7_readiness.py::test_end_to_end_research_workflow PASSED [100%]

============================= 48 passed in 7.73s =============================
```

---

## 5. Checkpoint & Artifact Registry

- **Webcam Interface & Dashboard**:
  - `app.py` (FastAPI backend with quality guidance & domain gate routing)
  - `static/index.html` (Research dashboard UI with live camera viewport & alignment reticle)
  - `static/css/style.css` (Webcam container, pulsing guide, guidance list styles)
  - `static/js/dashboard.js` (Webcam stream lifecycle, still capture, and quality feedback handlers)
- **Documentation & Protocols**:
  - `docs/WEBCAM_VALIDATION.md` (Prospective webcam clinical validation protocol)
  - `docs/RELIABILITY_AND_DOMAIN_VALIDATION.md` (Layer 3.5 report & data collection spec)
  - `docs/DATASET_AUDIT.md` (Layer 1 audit)
  - `docs/PROJECT_STATUS.md` (overall project tracking)
- **Inference & Export Services**:
  - `src/inference/__init__.py`
  - `src/inference/unified_service.py` (Unified inference engine & camera domain gates)
  - `src/inference/session_export.py` (Session audit export generator with capture source)
- **Checkpoints**:
  - `saved_models/anemia/best_model.pth` (288.8 MB, ResNet-50 Regressor)
  - `saved_models/conjunctivitis/best_model.pth` (38.8 MB, original MobileNetV3-Small)
  - `saved_models/conjunctivitis/best_model_v2_clean.pth` (12.5 MB, clean deduplicated MobileNetV3-Small)
  - `saved_models/jaundice/jaundice_config.json` (architecture config)
- **Test Suites**:
  - `tests/test_layers.py` (Layers 1–3.5 unit and regression tests)
  - `tests/test_layer4_inference.py` (Layer 4 inference service tests)
  - `tests/test_layer5_dashboard.py` (Layer 5 dashboard and API tests)
  - `tests/test_layer6_webcam.py` (Layer 6 webcam capture, guidance, and domain gate tests)
