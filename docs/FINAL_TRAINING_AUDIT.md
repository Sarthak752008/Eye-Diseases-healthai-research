# FINAL TRAINING AUDIT — EyeHealthAI

**Document Date**: 2026-10-12  
**Auditor**: Senior ML Researcher & Computer Vision Engineer  
**Status**: Pre-Training & Verification Complete  

---

## 1. Hardware & Compute Environment Inspection

Prior to any model execution or training, hardware resources and environment constraints were audited:

- **Operating System**: Windows (10 / 11)
- **CPU Cores**: 16 logical cores
- **Torch Threads**: 10 active worker threads
- **PyTorch Version**: 2.14.1+cpu
- **CUDA Acceleration**: **False** (`torch.cuda.is_available() == False`)
- **GPU Memory**: 0 MB (No NVIDIA CUDA GPU detected)
- **Cached Torch Hub Checkpoints**:
  - `resnet50-11ad3fa6.pth` (Pretrained ImageNet ResNet-50)
  - `mobilenet_v3_small-047dcff4.pth` (Pretrained ImageNet MobileNetV3-Small)
- **External Download Policy**: Offline / restricted sandbox environment; models must leverage locally cached pretrained backbones.

---

## 2. Model Pipeline & Checkpoint Audit

### 2.1 Model A: Anemia / Laboratory Hemoglobin Regression
- **Clinical Target**: Direct continuous prediction of laboratory venous hemoglobin in $g/dL$ ($Hb \in [7.0, 17.4]$).
- **Dataset**: `dataset anemia/India` (95 subjects, 382 multi-view crops) and `dataset anemia/Italy` (123 subjects, 480 multi-view crops). Total: 862 image-view records across 218 independent subjects.
- **Reference Ground Truth**:
  - `India.xlsx`: 95 subjects, all verified laboratory Hb values.
  - `Italy.xlsx`: 123 subjects, 122 verified laboratory Hb values (European commas converted). Subject #93 correctly flagged as unrecorded (`_`).
- **Partitioning Strategy (Zero-Leakage Guarantee)**:
  - Partitioned strictly at the **subject/patient level** stratified across 4 Hb quartiles.
  - **Train**: 151 subjects (601 views)
  - **Validation**: 33 subjects (131 views)
  - **Held-Out Test**: 33 subjects (130 views)
  - **Unlabeled**: 1 subject (Italian #93)
  - Subject overlap between partitions: **0%** ($P(\text{overlap}) = 0$).
- **Checkpoint Status**: `saved_models/anemia/best_model.pth` (288.8 MB).
  - Architecture: `AnemiaRegressor` (ResNet-50 backbone + 2-layer MLP projection head with Dropout).
  - Contains trained floating-point weights: 322 parameter tensors, mean weight norm = 43.55.
  - Training completed: Yes, early-stopped at epoch 4 on validation MAE (1.4530 g/dL).
- **Preprocessing Contract**:
  - Truncated image safe loading (`LOAD_TRUNCATED_IMAGES=True`).
  - RGB color conversion (no OpenCV BGR inverted channel bug).
  - Bilinear aspect-ratio preserving downsampling with centered black padding to $(224 \times 224)$.
  - ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
  - Max absolute tensor difference between training evaluation and inference preprocessing: **0.000000** (strictly identical).
- **Inference Determinism**: 3 consecutive runs on the same real test image yielded identical prediction: `[12.09, 12.09, 12.09] g/dL`.
- **Baseline Comparison (Held-Out Test Set N=33 Subjects, 130 Views)**:
  - *Train Mean Baseline* ($\hat{y} = 12.79$): Patient MAE = **1.988 g/dL**, Patient RMSE = **2.449 g/dL**.
  - *Train Median Baseline* ($\hat{y} = 13.00$): Patient MAE = **1.970 g/dL**, Patient RMSE = **2.459 g/dL**.
  - *Candidate MobileNetV3 Regressor*: Patient MAE = **2.983 g/dL** (inferior to naive baselines; rejected).
  - *ResNet-50 Regressor*: Patient MAE = **1.501 g/dL**, Patient RMSE = **2.073 g/dL**, Pearson $r = \mathbf{+0.518}$, $R^2 = \mathbf{+0.284}$.
  - The model demonstrates genuine clinical predictive signal, outperforming naive central tendency baselines by ~0.48 g/dL.

---

### 2.2 Model B: Conjunctival Redness / Injection
- **Clinical Target**: Binary classification of visible conjunctival injection (hyperemia) vs healthy anterior eye.
- **Dataset Audit & Resolution**:
  - *Mendeley Eye Diseases v2 Confound*: Audit proved 100% of Normal images are small square crops ($56 \times 56$ px) and 100% of Conjunctivitis images are macro shots ($352 \times 252$ px). A non-visual rule (`width > 100`) achieves 100% test accuracy without inspecting pixels. Gated permanently from diagnostic exposure.
  - *Clinical SLID Dataset*: 2,617 clinical slit-lamp photographs from Zhejiang University (*Frontiers in Digital Health*, 2025, CC BY 4.0). Both normal controls and injection cases originate from standardized high-resolution slit-lamp sensors ($1924 \times 1556$ px), eliminating the spatial frequency shortcut.
- **Data Inventory**: 552 high-resolution clinical slit-lamp images (307 Conjunctival Injection, 245 Zero-Lesion Controls).
- **Partitioning**: Train: 386 images | Validation: 84 images | Held-Out Test: 82 images. Zero cross-split MD5 collision.
- **Checkpoint Status**: `saved_models/conjunctivitis/slid_injection_model.pth` (13.1 MB, MobileNetV3-Small).
- **Empirical Baseline Comparison on SLID ($N=82$ Test Images)**:
  - *Metadata Baseline (Logistic Regression on Dimensions)*: Test Accuracy = **0.5732** (near chance).
  - *Metadata Baseline (Decision Tree on Dimensions)*: Test Accuracy = **0.5732**.
  - *MobileNetV3-Small CNN*: Test Accuracy = **0.9878** (81/82), Sensitivity = **0.9783** (45/46), Specificity = **1.0000** (36/36), F1-Score = **0.9890**, ROC-AUC = **1.0000**.
  - Confusion Matrix: `[[36, 0], [1, 45]]`.
- **Scope & Safety**: Explicitly designated as a research screening tool for visible conjunctival hyperemia. Does not identify microbial or allergic etiology.

---

### 2.3 Model C: Jaundice / Visible Scleral Yellowing
- **Clinical Target**: Binary categorical screening of visible scleral icterus (normal vs jaundiced sclera).
- **Dataset**: *Diverse Jaundice Eye Dataset* (Daniel Ekong, Hugging Face, CC BY-NC 4.0).
- **Missing Clinical Lab Ground Truth**:
  - UCL Cirrhosis study (`10.5522/04/24083487.v1`) holds raw patient images under NHS Research Ethics Committee approval (REC Ref: 08/H0714/8). Only summary Bland-Altman spreadsheets are open; raw patient photos are restricted.
  - In strict compliance with `AGENTS.md` clinical safety standards, **zero synthetic serum bilirubin values are fabricated**. Numerical bilirubin estimation is permanently gated.
- **Data Inventory**: 321 ocular photographs (176 healthy, 145 visibly jaundiced across Fitzpatrick skin phototypes).
- **Partitioning**: Train: 256 images (141 healthy, 115 jaundice) | Held-Out Test: 65 images (35 healthy, 30 jaundice).
- **Checkpoint Status**: `saved_models/jaundice/best_model.pth` (13.1 MB, MobileNetV3-Small).
- **Test Metrics ($N=65$ Test Images)**:
  - *Majority Baseline*: Accuracy = **0.5385**, Sensitivity = **0.0000**.
  - *MobileNetV3-Small CNN*: Accuracy = **0.8000** (52/65), Precision = **0.9048**, Specificity = **0.9429** (33/35), Sensitivity = **0.6333** (19/30), F1-Score = **0.7451**, ROC-AUC = **0.9238**.
  - Confusion Matrix: `[[33, 2], [11, 19]]`.
- **Licensing Constraint**: CC BY-NC 4.0 restricts usage to non-commercial academic research.

---

## 3. Pre-Training Audit Summary & Action Plan

1. **Model A (Anemia)**: ResNet-50 regressor is verified to be significantly superior to central tendency baselines and lightweight MobileNet regressor. Retain ResNet-50 as the primary primary model and ensure multi-view aggregation is active.
2. **Model B (Conjunctival Injection)**: SLID clinical slit-lamp dataset completely resolves the Mendeley metadata shortcut. Model B is operational as a research tool for visible injection.
3. **Model C (Jaundice)**: Operates strictly as a categorical scleral icterus screening research tool. Numerical serum bilirubin remains locked to prevent fabricated predictions.
4. **End-to-End Application**: Verify that new-image inference works seamlessly without caching, that UI cards and two-page PDF reports display honest, current-session metrics without placeholders, and that all 106 automated tests continue passing.
