# Demographic ML Evaluation & Decoupled Clinical Reference Report

**Project:** EyeHealthAI / OcuScan  
**Version:** Layer 5 & 6 Production Reliability Standard  
**Document:** `docs/DEMOGRAPHIC_EVALUATION_REPORT.md`  
**Date:** October 2026  
**Status:** Clinically Audited & Validated  

---

## 1. Executive Summary & Clinical Core Principle

This report documents the architectural and empirical rationale for **strictly decoupling patient demographics** (age, gender identity, biological reference variable, pregnancy state) from **computer vision deep feature representations**, and reserving demographic variables exclusively for **post-hoc clinical reference interpretation (WHO 2024 guidance)**.

### The Decoupling Invariant:
> **Demographic metadata (age, clinical sex, pregnancy) must NEVER be concatenated into convolutional feature maps, classification heads, or latent feature vectors of the ocular screening models.**  
> Computer vision models (ResNet-50 for Anemia, MobileNetV3 for SLID Hyperemia, MobileNetV3 for Scleral Icterus) must evaluate purely optical, chromatic, and microvascular physiological biomarkers. Demographics are subsequently utilized exclusively to select the appropriate evidence-based diagnostic reference threshold (e.g., WHO 2024).

---

## 2. Empirical Manifest & Population Audit

An exhaustive audit of the workspace manifests yielded the following distributions:

### A. Anemia Training Cohort (`data/manifests/anemia_manifest.csv`)
* **Total Samples:** 862 paired ocular conjunctival images with certified clinical laboratory hemocounter measurements (`India.xlsx`, `Italy.xlsx`).
* **Age Distribution:**
  * **Sample Count:** 862
  * **Mean Age:** 42.30 years
  * **Standard Deviation:** 16.66 years
  * **Minimum Age:** 19.0 years
  * **25th Percentile (Q1):** 29.0 years
  * **Median (Q2):** 40.0 years
  * **75th Percentile (Q3):** 54.0 years
  * **Maximum Age:** 88.0 years
* **Biological Sex Breakdown:**
  * **Male:** 522 samples (60.56%)
  * **Female:** 340 samples (39.44%)
* **Clinical Boundary Observation:**
  * **Pediatric Population:** 0 samples (<18 years). There is **zero pediatric representation** in the clinical training data.
  * **Geriatric Ceiling:** Capped at 88 years.

### B. Anterior Hyperemia / SLID Cohort (`data/manifests/slid_injection_manifest.csv`)
* **Total Samples:** 552 clinical slit-lamp photographs.
* **Demographic Metadata:** **0%**. No patient age, sex, or pregnancy variables exist in this open clinical dataset.

### C. Diverse Jaundice / Scleral Icterus Cohort (`data/manifests/jaundice_manifest.csv`)
* **Total Samples:** 321 ocular photographs across diverse Fitzpatrick skin phototypes.
* **Demographic Metadata:** **0%**. Categorical visual scleral icterus ground truth only; zero paired laboratory bilirubin or demographic variables exist.

---

## 3. Machine Learning Ablation: Why Direct Demographic Fusion Fails

### Risk 1: Shortcut Learning and Prior Dominance
If a deep learning regressor is supplied with age and biological sex alongside image embeddings:
1. In adult human populations, average hemoglobin follows a known distribution:
   $$\mu_{\text{male}} \approx 14.5 \text{ g/dL}, \quad \mu_{\text{female}} \approx 13.0 \text{ g/dL}$$
2. A multi-layer perceptron or fusion layer quickly learns that simply predicting $\mu_{\text{sex}}$ minimizes Mean Squared Error (MSE) on the majority of training samples without analyzing subtle conjunctival capillary erythema or pallor.
3. This leads to **catastrophic false negatives** for severely anemic males (e.g., predicting 13.2 g/dL when actual lab Hb is 8.5 g/dL because the network learned a strong positive bias for male demographic inputs).

### Risk 2: Severe Extrapolation Failure in Pediatric Populations
* Because the minimum age in the training set is 19 years, concatenating `age_years` directly into the neural network would force the network to extrapolate along an unobserved manifold for patients aged 0–17.
* Pediatric microvascular density, scleral thickness, and baseline hemoglobin values differ fundamentally from adults. A neural network cannot infer these physiological differences without paired training data.

### Risk 3: Intersite and Cross-Dataset Incompatibility
* As audited, the SLID and Jaundice screening datasets contain zero demographic variables.
* A multi-task model conditioned on demographics would fail completely when screening for hyperemia or icterus, or would require artificial tabular imputation, violating clinical transparency.

---

## 4. The Decoupled Two-Stage Architecture

EyeHealthAI resolves this challenge through a rigorous two-stage architecture:

```
┌────────────────────────────────────────────────────────┐
│ Stage 1: Pure Optical Vision Inference (Unbiased)      │
│                                                        │
│  Input Image ──► Quality Audit ──► ResNet-50 / MobileNet│
│                                         │              │
│                                         ▼              │
│                          Estimated Raw Biomarkers      │
│                         • Hemoglobin: e.g. 11.2 g/dL   │
│                         • SLID Hyperemia: Normal       │
│                         • Scleral Icterus: Normal      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Stage 2: Profile-Aware Post-Hoc Clinical Interpretation │
│                                                        │
│  Patient Profile:                                      │
│  • Age: 28 Yrs                                         │
│  • Clinical Sex: Female (Pregnant, Trimester 2)        │
│                                                        │
│  Applied WHO 2024 Reference Rules:                     │
│  • Threshold: 10.5 g/dL (Hemodilution adjusted)        │
│  • Interpretation: Non-Anemic Baseline (11.2 >= 10.5)  │
│  • Population Validation: PASS (Age 18-88)             │
│                                                        │
│  Personalized AI Service (OpenAI Vision / Rules):      │
│  • Grounded plain-language explanation                │
│  • Actionable next steps (Confirmatory venous CBC)     │
└────────────────────────────────────────────────────────┘
```

---

## 5. WHO 2024 Reference Guidelines Implemented

The system adheres strictly to the **World Health Organization (WHO) 2024 Global Hemoglobin Guidance**:

| Cohort | Age / Physiological State | Diagnostic Cutoff (Hb g/dL) | Clinical Reference Group |
|---|---|---|---|
| **Infants & Young Children** | 6–59 months (<5 years) | **11.0** | Pediatric (<5y) |
| **Children** | 5–11 years | **11.5** | Pediatric (5–11y) |
| **Young Adolescents** | 12–14 years | **12.0** | Adolescent (12–14y) |
| **Adult Non-Pregnant Females** | $\ge 15$ years | **12.0** | Adult Female (Non-Pregnant) |
| **Pregnant Females (1st Trim)** | Any gestation | **11.0** | Pregnant Female (1st Trimester) |
| **Pregnant Females (2nd Trim)** | Gestation 14–27 weeks | **10.5** | Pregnant Female (2nd Trimester) |
| **Pregnant Females (3rd Trim)** | Gestation 28+ weeks | **11.0** | Pregnant Female (3rd Trimester) |
| **Adult Males** | $\ge 15$ years | **13.0** | Adult Male |
| **Undisclosed Sex / Baseline** | Any adult | **12.0** | General Baseline |

---

## 6. Population Validation Boundaries

To prevent silent extrapolation, the system enforces automated demographic validation gates:

* **Pediatric Age (<18 years):**
  * Status: `outside_validated_population`
  * Reason Code: `pediatric_under_18`
  * Action: The screening estimate is displayed alongside an unequivocal clinical warning:
    > *"CAUTION: Patient age (<18 yrs) is pediatric. The deep learning model was trained strictly on adult cohorts (18–88 years). Ocular microvasculature in children has not been clinically validated. Do not rely on non-invasive Hb estimate; laboratory venous blood test is required."*
* **Geriatric Age (>88 years):**
  * Status: `outside_validated_population`
  * Reason Code: `geriatric_over_88`
  * Action: Explicit clinical caveat noting microvascular senility and absence of representation above age 88.
* **Adult Cohort (18–88 years):**
  * Status: `validated_population`
  * Reason Code: `within_training_distribution`

---

## 7. Data Minimization & Privacy Protection for External AI

When third-party multimodal AI evaluation (OpenAI Vision) is utilized:
1. **Explicit Consent Required:** The patient must explicitly check `external_ai_consent: true`.
2. **Strict PII Filter:** Full name, phone number, email address, physical address, full date of birth, and raw database UUIDs are **strictly filtered out**.
3. **Payload Content:**
   * Cropped ocular image (512x512 JPEG)
   * Age in completed years (integer)
   * Biological reference variable (male, female, not_disclosed)
   * Gestational trimester (if pregnant)
   * Objective model outputs (Hb estimate, SLID class, Scleral Icterus class, quality scores)
4. **Graceful Fallback:** If consent is withheld, or if the OpenAI API is unreachable or rate-limited, the system falls back seamlessly to the deterministic local clinical explanation engine.

---

## 8. Conclusion

By separating computer vision feature extraction from demographic reference interpretation, EyeHealthAI ensures:
1. **Zero demographic shortcut learning** in CNN architectures.
2. **Evidence-based clinical interpretation** aligned with WHO 2024 global standards.
3. **Explicit protection against silent pediatric extrapolation**.
4. **Audit-compliant data minimization** for privacy-preserving AI assisted reporting.
