# Phase 5 — Model Comparison & Retraining Report

**Document Date**: 2026-10-11  
**Auditor & Research Engineer**: Senior ML Engineer & CV Researcher  
**Hardware Environment**: 16-Core Intel CPU, 32 GB RAM, CUDA Acceleration: False  

---

## 1. Executive Summary & Verification Matrix

All models were evaluated on strictly held-out test partitions with zero subject or duplicate data leakage.

| Clinical Module | Evaluated Architecture | Dataset & Source | Ground-Truth Target | Held-Out Test Metric | Baseline Comparison | Selected Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Anemia** | **ResNet-50 Regressor** | Eyes-defy-anemia (India & Italy cohorts, $N=218$ subjects) | Laboratory Venous Hemoglobin ($g/dL$) | **Patient MAE: 1.5013 g/dL**<br>Patient RMSE: 2.0728 g/dL<br>Pearson $r$: 0.518, $R^2$: 0.284 | Beats Mean Baseline (1.988 g/dL) & Median Baseline (1.970 g/dL) | **SELECTED PRIMARY MODEL** (Validated on validation set MAE: 1.4530 g/dL) |
| **Model A Candidate** | MobileNetV3-Small Regressor | Eyes-defy-anemia | Laboratory Venous Hemoglobin ($g/dL$) | Patient MAE: 2.9826 g/dL<br>Patient RMSE: 3.5110 g/dL | Inferior to Mean & Median Baselines | Not selected (Failed validation gate: Val MAE 2.4721 g/dL) |
| **Model B: Confounded** | MobileNetV3-Small Classifier | Mendeley Eye Diseases v2 ($N=965$ deduplicated) | Categorical Conjunctivitis | Test Accuracy: 1.0000 | **Metadata-Only Rule (`width > 100`): 1.0000** | **AUDITED & BLOCKED** (Artifact of 56px vs 352px image size shortcut) |
| **Model B: Clinical SLID** | **MobileNetV3-Small Classifier** | SLID Dataset (Zhejiang University, Frontiers in Digital Health 2025, $N=552$) | Clinical Slit-Lamp Conjunctival Injection | **Test Accuracy: 0.9878**<br>Sensitivity: 0.9783<br>Specificity: 1.0000<br>F1: 0.9890, ROC-AUC: 1.0000 | Metadata Baselines: Logistic Regression: 0.5732, Decision Tree: 0.5732 (No shortcut) | **SELECTED CLINICAL INJECTION MODEL** (Slit-lamp research screening only) |
| **Model C: Jaundice** | **MobileNetV3-Small Classifier** | Diverse Jaundice Eye Dataset (Daniel Ekong, CC BY-NC 4.0, $N=321$) | Categorical Visible Scleral Icterus | **Test Accuracy: 0.8000**<br>Precision: 0.9048<br>Specificity: 0.9429<br>Sensitivity: 0.6333<br>F1: 0.7451, ROC-AUC: 0.9238 | Majority Class Baseline: 0.5385 | **SELECTED RESEARCH MODEL** (Categorical icterus only; Non-commercial; No lab bilirubin) |

---

## 2. Model A: Anemia / Laboratory Hemoglobin Regression

### 2.1 Dataset Partitioning & Leakage Prevention
- **Cohort**: 862 multi-view images across 218 independent subjects (95 India, 123 Italy).
- **Stratified Partitioning**: Partitioned strictly at the **subject level** across 4 hemoglobin quartiles:
  - **Train**: 151 subjects (601 multi-view images).
  - **Validation**: 33 subjects (131 multi-view images).
  - **Held-Out Test**: 33 subjects (130 multi-view images).
- All multi-view images (`palpebral`, `forniceal`, `forniceal_palpebral`, `raw_eye`) belonging to the same subject reside strictly in the same split ($P(\text{overlap}) = 0$).

### 2.2 Empirical Comparison vs Baselines
Evaluated on the held-out test split (33 patients, 130 views):

| Model / Approach | Parameters | Validation Patient MAE (g/dL) | Test Patient MAE (g/dL) | Test Patient RMSE (g/dL) | Test View MAE (g/dL) | Test View RMSE (g/dL) | Pearson $r$ | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Baseline** ($\hat{y} = 12.79$) | 0 | 1.892 | 1.988 | 2.449 | 2.008 | 2.466 | 0.000 | 0.000 |
| **Median Baseline** ($\hat{y} = 13.00$) | 0 | 1.874 | 1.970 | 2.459 | 1.992 | 2.477 | 0.000 | -0.008 |
| **MobileNetV3-Small Regressor** | 1.8M | 2.472 | 2.983 | 3.511 | 3.125 | 3.682 | +0.124 | -0.682 |
| **ResNet-50 Regressor** | 24.1M | **1.453** | **1.501** | **2.073** | **1.784** | **2.285** | **+0.518** | **+0.284** |

### 2.3 Model Selection Decision
- **Chosen Model**: `ResNet-50 Regressor` (`saved_models/anemia/best_model.pth`).
- **Rationale**: Chosen based strictly on **validation set MAE** (1.4530 g/dL vs 2.4721 g/dL for MobileNetV3). ResNet-50 is the only architecture that outperforms central tendency baselines by a significant margin (0.49 g/dL lower MAE than the mean baseline).
- **Session Aggregation Impact**: Single-view test MAE is 1.78 g/dL; aggregating multiple views from the same session lowers the MAE to 1.50 g/dL, proving the clinical value of multi-view capture.
- **Limitation**: An MAE of ~1.5 g/dL means individual predictions carry uncertainty ($\pm 1.5$ g/dL). It is suitable for non-invasive risk screening, NOT definitive diagnostic replacement for venous blood draw.

---

## 3. Model B: Conjunctival Redness / Injection

### 3.1 Resolution Shortcut on Mendeley Dataset
The Mendeley dataset (`Image Dataset on Eye Diseases Classification...`) exhibited severe structural bias:
- Normal class: Mean width $56.16$ px (tight iris/pupil crops).
- Conjunctivitis class: Mean width $352.47$ px (full-eye shots).
- A non-visual rule (`width > 100`) achieves 100% test accuracy without inspecting pixels. The deep model learned resampling fingerprints rather than conjunctival hyperemia.

### 3.2 Clinical Resolution on SLID Dataset
To resolve this shortcut, we acquired the **SLID (Slit-Lamp Image Dataset)** from Zhejiang University (Mingyu Xu et al., Frontiers in Digital Health 2025):
- 2,617 clinical slit-lamp photographs at $1924 \times 1556$ or $2576 \times 1934$ resolution.
- Extracted 552 images: 307 with verified `Conjunctival injection`, 245 healthy anterior-segment controls.
- Metadata baselines (Logistic Regression & Decision Tree) achieved **0.5732 test accuracy** (near random chance), proving that image dimensions cannot solve the task on SLID.
- Trained MobileNetV3-Small with color-preserving augmentations:
  - **Held-Out Test Accuracy**: **0.9878** (81/82 correct).
  - **Sensitivity**: **0.9783** (45/46 injection cases detected).
  - **Specificity**: **1.0000** (36/36 normal controls detected).
  - **F1-Score**: **0.9890**, **ROC-AUC**: **1.0000**.
  - **Confusion Matrix**: `[[36, 0], [1, 45]]`.
- **Clinical Scope**: Validated for **Visible Conjunctival Injection / Hyperemia** on anterior-segment macro/slit-lamp photography. Explicitly disclaims identifying specific bacterial, viral, or allergic etiology.

---

## 4. Model C: Jaundice / Visible Scleral Yellowing

### 4.1 Ground-Truth Status & Ethical Boundaries
- Zero clinical datasets with paired laboratory serum bilirubin are publicly accessible for raw image training (the UCL cirrhosis dataset distributes only Bland-Altman summary tables; raw patient images are restricted under NHS ethics).
- In accordance with our safety rules, **zero synthetic bilirubin values were fabricated**.
- Numerical bilirubin estimation is strictly locked.

### 4.2 Categorical Scleral Icterus Classification
We trained and evaluated `JaundiceClassifier` on the **Diverse Jaundice Eye Dataset** (Daniel Ekong, Hugging Face, CC BY-NC 4.0):
- 321 web images (176 normal, 145 visibly jaundiced sclera across diverse skin phototypes).
- Preprocessing strictly preserved scleral yellow chromaticity (zero hue shifts).
- **Held-Out Test Results** (N=65: 35 Normal, 30 Jaundice):
  - **Test Accuracy**: **0.8000** (52/65).
  - **Precision**: **0.9048** (19/21 positive predictions correct).
  - **Specificity**: **0.9429** (33/35 healthy eyes correctly rejected).
  - **Sensitivity**: **0.6333** (19/30 visibly jaundiced eyes identified).
  - **F1-Score**: **0.7451**.
  - **ROC-AUC**: **0.9238**.
  - **Confusion Matrix**: `[[33, 2], [11, 19]]`.
- **Status**: Exists as an operational, research-only screening tool for visible scleral yellowing. Blocked from commercial use under CC BY-NC 4.0 terms.
