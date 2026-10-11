# Layer 3.5 — Reliability and Domain Validation Report

**Project**: EyeHealthAI  
**Target Milestone**: Domain Generalization, Confounding Audits, and Clinical Specification  
**Date**: 2026-10-11  

---

## 1. Conjunctivitis Dataset Audit & Metadata-Only Baseline

### 1.1 Empirical Dataset Audit
An exhaustive audit of `data/manifests/conjunctivitis_manifest_v2_clean.csv` (965 deduplicated images) reveals severe structural asymmetries between classes:

| Class | Sample Count | Width (Min / Max / Mean) | Height (Min / Max / Mean) | Aspect Ratio (Min / Max / Mean) | Area in Pixels (Min / Max / Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 649 | 41 / 89 / **56.16** px | 41 / 89 / **56.16** px | 1.000 / 1.000 / **1.000** | 1,681 / 7,921 / **3,208.8** |
| **Conjunctivitis** | 316 | 117 / 741 / **352.47** px | 120 / 557 / **252.55** px | 0.821 / 2.398 / **1.403** | 14,274 / 359,385 / **90,992.6** |

### 1.2 Metadata-Only Baseline Performance
To evaluate whether class labels could be predicted without any biological image content, we evaluated three non-visual metadata classifiers trained solely on `[width, height, aspect_ratio, area]` across the exact same untouched test split (N=146: 98 Normal, 48 Conjunctivitis):

| Model | Input Features | Test Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule: `width > 100`** | Width only | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[98, 0], [0, 48]]` |
| **Logistic Regression** | Width, Height, Aspect, Area | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[98, 0], [0, 48]]` |
| **Decision Tree (Depth=1)** | Area threshold split | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[98, 0], [0, 48]]` |
| **MobileNetV3-Small (CNN)** | 224×224 RGB Image Pixels | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[98, 0], [0, 48]]` |

**Empirical Conclusion**:
The metadata-only models achieve **identical 100% test performance** to the deep convolutional network. Because image dimensions and aspect ratios are completely linearly separable, the deep network does not need to learn conjunctival injection, vascular engorgement, or discharge—it can achieve 100% test accuracy entirely via resampling and high-frequency interpolation artifacts.

---

## 2. Why Resizing & Standard Preprocessing Cannot Solve the Confounding

A standard engineering attempt to address resolution differences is resizing all inputs to a uniform dimension (e.g. 224×224). **Resizing does not solve this confounding**, for three physiological and mathematical reasons:

1. **Nyquist-Shannon & Resampling Fingerprints**:
   - An image originally 56×56 upsampled to 224×224 has an effective spatial bandwidth capped at 28 cycles/image. Bicubic/bilinear interpolation introduces distinct smoothness and ringing patterns in the Fourier domain.
   - An image originally 350×250 downsampled to 224×224 retains high-frequency edge gradients and camera sensor noise.
   - CNN convolutional kernels act as high-pass spatial filters that easily detect the presence or absence of these interpolation spectral signatures.
2. **Framing & Anatomical Divergence**:
   - The Normal class images are tightly cropped square sub-regions (iris or partial scleral patches).
   - The Conjunctivitis images are macro, whole-eye captures displaying eyelids, eyelashes, medial/lateral canthi, and surrounding periorbital skin.
   - Resizing changes pixel dimensions, but does not harmonize anatomical framing.
3. **Clinical Color Preservation Constraint**:
   - Aggressive augmentations (e.g., severe color jitter, histogram equalization, random blur, or heavy Gaussian noise) that might disrupt spectral shortcuts are prohibited by clinical safety rules: conjunctival erythema is fundamentally an optical chromatic signal. Heavy artificial noise corrupts genuine clinical diagnostic features.

---

## 3. Data Collection Specification: Clinical Conjunctivitis & Normal Controls

To develop a clinically valid Model B, new data must be collected following this rigorous specification:

### 3.1 Device & Optical Hardware Standard
- **Primary Imaging Device**: Standardized smartphone hardware (e.g., iPhone 13+ or Samsung Galaxy S22+) mounted with a dedicated clinical anterior-segment macro adapter (or digital slit-lamp camera).
- **Camera Configuration**:
  - Fixed focal distance: 10–15 cm from ocular surface.
  - Locked white balance: 5500K preset (no automatic dynamic tone-mapping or vendor "beauty/smoothing" filters).
  - Exposure: Fixed manual exposure calibrated against an integrated 18% neutral gray card or standard 24-patch X-Rite ColorChecker.
  - Lighting: Cross-polarized diffuse ring illuminator (CRI ≥ 95, 5000K–5500K) to eliminate corneal specular glints while preserving scleral/conjunctival chromatic saturation.

### 3.2 Anatomical Framing & Acquisition Protocol
For each patient session, three standardized anatomical views must be captured identically for both healthy controls and symptomatic subjects:
1. **Inferior Palpebral Conjunctiva**: Gentle manual eversion of the lower eyelid while patient directs gaze upward.
2. **Bulbar Conjunctiva (Nasal & Temporal)**: Wide-angle open eye with patient directing gaze extreme left then extreme right.
3. **Full Anterior Eye Overview**: Neutral primary gaze showing palpebral fissure, limbus, and sclera.

### 3.3 Cohort & Clinical Ground Truth Definition
- **Case Definition (Conjunctivitis)**:
  - In-person evaluation by a board-certified ophthalmologist or optometrist.
  - Grading on validated clinical scales: Brien Holden Vision Institute (BHVI) Hyperemia Scale (Grade 0–4) or Efron Scale.
  - Clinical subtype documentation: Bacterial (purulent discharge), Viral (follicular reaction, watery discharge), Allergic (papillary reaction, itching), or Toxic/Non-infectious.
- **Control Definition (Normal Healthy)**:
  - Age- and sex-matched controls without active ocular surface disease, contact lens-induced hyperemia, dry eye disease (OSDI < 13), or ocular surgery in the past 6 months.
- **Strict Anti-Fabrication Rule**:
  - No synthetic images, generative diffusion samples, or artificial recoloring. All images must originate from documented clinical encounters with ethical IRB approval.

---

## 4. Source-Independent Evaluation Plan

Future validation must use a multi-center, source-independent partitioning strategy:

1. **Center / Institution-Level Splitting**:
   - Training images from Clinic A; Validation images from Clinic B; Test images from Clinic C.
   - This ensures the model cannot memorize background clinic lighting, wall paint reflectance, or camera sensor fixed pattern noise.
2. **Patient-Level Isolation**:
   - Strict zero patient overlap between train, val, and test partitions ($P(\text{overlap}) = 0$).
3. **Out-of-Distribution Robustness Stress Tests**:
   - Subgroup analysis across Fitzpatrick skin phototypes (I through VI).
   - Robustness under slight defocus ($\pm 0.5$ diopter) and ambient light shifts (3000K incandescent vs 6500K daylight).

---

## 5. Anemia Model (Model A) Reliability Audit & Current Standing

### 5.1 Retained Checkpoint & Validation Status
The existing ResNet-50 regressor checkpoint (`saved_models/anemia/best_model.pth`) and validation audit (`saved_models/anemia/validation_report.json`) remain fully intact and verified.

| Split | Level | ResNet-50 MAE (g/dL) | ResNet-50 RMSE (g/dL) | Mean Baseline MAE (g/dL) | Median Baseline MAE (g/dL) | ResNet-50 $R^2$ | Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** | View Level | **1.5933** | 2.1475 | 2.0931 | 2.0500 | 0.2287 | 0.5547 |
| **Validation** | Patient Level | **1.4530** | 1.9936 | 2.0931 | 2.0500 | **0.3353** | **0.6103** |
| **Test** | View Level | **1.7757** | 2.2915 | 2.0099 | 1.9877 | 0.1367 | 0.4645 |
| **Test** | Patient Level | **1.5013** | 2.0728 | 1.9895 | 1.9606 | **0.2836** | **0.5529** |

### 5.2 Documented Model Limitations
1. **Clinical Error Tolerance**:
   - Clinical WHO triage thresholds for anemia in adults are strict: $<11.0$ g/dL (pregnant), $<12.0$ g/dL (females), $<13.0$ g/dL (males).
   - An MAE of $\approx 1.50$ g/dL means a patient with true Hb of 11.5 g/dL could easily be predicted as 13.0 g/dL (false negative) or 10.0 g/dL (false positive).
   - Model A provides evidence of non-invasive optical correlation ($r \approx 0.55$–$0.61$), but is **not clinically suitable for stand-alone diagnostic decision-making**.
2. **Multi-View Intra-Patient Variability**:
   - Within the same patient session, predictions across different anatomical crops (`forniceal`, `palpebral`, `forniceal_palpebral`) vary with an average standard deviation of **0.97 g/dL** (and maximum of 2.41 g/dL).
   - Aggregating multiple views via patient-level averaging improves $R^2$ (from 0.14 to 0.28) and lowers MAE (from 1.78 to 1.50 g/dL).
3. **No Retraining Justification**:
   - The current ResNet-50 model extracts the maximum reliable signal achievable from the current 218-subject dataset without synthetic data fabrication or label leakage. Retraining without additional clinical cohorts would risk overfitting.

---

## 6. Integration Readiness Gates (Conditions Before Downstream Use)

| Model Target | Current Architecture | Current Status | Integration Readiness Gate (Requirements to Advance) |
| :--- | :--- | :--- | :--- |
| **Model A: Anemia** | ResNet-50 Regressor | Validated research baseline (Patient Test MAE: 1.50 g/dL) | **CONDITIONAL ADVANCE**: May be integrated as an *investigational assistive indicator* with mandatory disclaimer and uncertainty display. Must average all available views for a subject. Cannot be presented as a definitive diagnostic tool. |
| **Model B: Conjunctivitis** | MobileNetV3-Small | Remediated & Audited; confounded by dataset framing/resolution | **BLOCKED FROM CLINICAL TRUST**: Although clean checkpoint runs with 100% test accuracy, metadata baseline proves the task is solvable without pixels. Model B cannot be trusted for clinical deployment until validated on the Section 3 matched dataset. |
| **Model C: Jaundice** | MobileNetV3-Small | Scaffolded & Forward-Pass Verified | **BLOCKED BY MISSING DATA**: Zero dataset exists in workspace. Integration blocked until genuine clinical scleral images with matched serum bilirubin are ingested. |
