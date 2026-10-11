# Layer 1 Dataset Audit Report (Updated & Remediated)

**Audit Date**: 2026-10-11  
**Workspace**: `c:\Users\sarth\OneDrive\Pictures\Innotech`  
**Auditor**: Senior ML Engineer & CV Researcher  

---

## Executive Summary

This audit provides an exhaustive, clinically grounded inspection of ocular datasets within the workspace and acquired external clinical sources. No clinical labels or patient identities were inferred from ambiguous filenames.

| Dataset Target | Source Directories / Repositories | Total Files / Images | Verified Labels | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model A: Anemia (Hb Regression)** | `dataset anemia/India`, `dataset anemia/Italy` | 862 files (218 subjects) | 217 subjects with verified venous lab Hb (1 unrecorded) | **Verified & Ready (0% Leakage)** |
| **Model B: Conjunctivitis (Mendeley)** | `Image Dataset on Eye Diseases Classification...` | 1,006 images (965 clean unique) | Class folders (Web-scraped) | **Audited & Blocked** (100% resolution confounding: 56px vs 352px) |
| **Model B: Conjunctival Injection (SLID)** | `xumingyu-hub/SLID` (Frontiers in Digital Health 2025) | 2,617 clinical slit-lamp images | 307 Conjunctival Injection, 245 Zero-Lesion Controls | **Acquired & Verified** (High-res $1924\times 1556$ px, CC BY 4.0) |
| **Conjunctiva Segmentation** | `Eye Conjunctiva Segmentation Dataset` | 547 eye images + 547 paired masks | Anatomical segmentation masks (Annotators 1 & 2) | **Verified & Ready** |
| **Model C: Jaundice / Scleral Icterus** | `Bleachcarte/Jaundice_Dataset` (Hugging Face) | 321 web images | Categorical (176 Healthy, 145 Jaundice) | **Acquired & Verified** (CC BY-NC 4.0, Research Only; No Lab Bilirubin) |
| **Model C: Numerical Bilirubin (Lab)** | UCL Cirrhosis Study (`10.5522/04/24083487.v1`) | 0 raw images accessible | 0 raw image-to-bilirubin pairs | **Raw Images Restricted** (NHS IRB 08/H0714/8) |

---

## 1. Anemia Dataset (`Eyes-defy-anemia`)
- **Origin**: Clinical datasets from Bari, Italy and Chennai, India captured under calibrated LED illumination.
- **Reference Spreadsheets**:
  - `India/India.xlsx`: 95 subjects (all 95 have verified laboratory Hb values, range: 7.0 – 17.4 g/dL).
  - `Italy/Italy.xlsx`: 123 subjects. 122 have verified laboratory Hb values with European decimal commas correctly converted. Subject 93 has unrecorded Hb (`_`), as documented by dataset authors.
- **File Integrity & Truncated Header Handling**:
  - All 862 images are fully valid and non-corrupt when opened with PIL using truncated image handling (`LOAD_TRUNCATED_IMAGES=True`).
- **Duplicate Audit**:
  - Found 2 duplicate mask files in `India/8/` (`*(1).png`), which are exact MD5 duplicates and flagged in the manifest.
- **View Mapping**:
  - Each subject folder contains the raw eye image (`raw_eye`) and hand-segmented conjunctiva regions (`palpebral`, `forniceal`, `forniceal_palpebral`).
- **Data Leakage Prevention**:
  - Splitting was performed strictly at the **subject level** (stratified across 4 Hb quartiles):
    - **Train**: 151 subjects (601 views)
    - **Validation**: 33 subjects (131 views)
    - **Test**: 33 subjects (130 views)
    - **Unlabeled**: 1 subject (Italian #93)
  - All multi-view images of a patient are locked into the same partition ($P(\text{leakage}) = 0$).

---

## 2. Conjunctivitis Datasets

### 2.1 Mendeley Data v2 (`Image Dataset on Eye Diseases Classification...`)
- **Origin**: Abu Kowshir Bitto & Marzia Ahmed (Mendeley Data, CC BY 4.0).
- **Class Counts**: 357 Conjunctivitis, 649 Normal.
- **Confounding Audit**: Normal images have mean resolution $56 \times 56$ px (square crop of iris/pupil), while Conjunctivitis images have mean resolution $352 \times 252$ px (macro eye photo).
- **Status**: Gated in clinical inference. Even after deduplication, a non-visual metadata classifier achieves 100% accuracy based on dimensions alone.

### 2.2 SLID Dataset (Clinical Slit-Lamp Anterior Eye Photography)
- **Origin**: Mingyu Xu et al., Zhejiang University School of Medicine (Frontiers in Digital Health 2025, CC BY 4.0).
- **Total Images**: 2,617 high-resolution digital slit-lamp images ($1924 \times 1556$ or $2576 \times 1934$ px).
- **Verified Lesions**:
  - `Conjunctival injection`: 307 images.
  - `Zero Lesions (Normal Controls)`: 245 images.
- **Optical Harmonization**: Both classes originate from standardized digital slit lamps with matched high-resolution sensors and focal planes, eliminating the spatial frequency shortcut.
- **Scope**: Targets visible conjunctival injection/hyperemia. Does not establish microbial or allergic etiology.

---

## 3. Jaundice Datasets

### 3.1 Hugging Face `Bleachcarte/Jaundice_Dataset`
- **Origin**: Daniel Ekong (Diverse Jaundice Eye Dataset, CC BY-NC 4.0).
- **Sample Count**: 321 images total.
  - Train: 256 images (141 Healthy, 115 Jaundice).
  - Test: 65 images (35 Healthy, 30 Jaundice).
- **Licence Terms**: CC BY-NC 4.0 permits academic and non-commercial research; commercial use is restricted.
- **Clinical Capability**: Categorical visible scleral icterus screening only. No numerical serum bilirubin ground truth exists in the dataset.

### 3.2 Clinical Bilirubin Ground Truth (UCL Cirrhosis Cohort)
- **Status**: Raw facial/ocular photographs are legally restricted under NHS ethics (London - Harrow REC Ref: 08/H0714/8). Only processed regression tables are published in the open repository.
- **Safety Policy**: Direct numerical bilirubin regression ($mg/dL$ or $\mu mol/L$) remains locked to prevent fabricated predictions.
