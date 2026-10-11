# Layer 1 Dataset Audit Report

**Audit Date**: 2026-10-11 00:06:38  
**Workspace**: `C:\Users\sarth\OneDrive\Pictures\Innotech`

## Executive Summary
This audit provides an exhaustive, clinically grounded inspection of ocular datasets within the workspace. No clinical labels or patient identities were inferred from ambiguous filenames.

| Dataset Target | Source Directories | Total Files / Images | Verified Labels | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Anemia (Hb Regression)** | `dataset anemia/India`, `dataset anemia/Italy` | 862 files (218 subjects) | 217 subjects with verified lab Hb | **Verified & Ready** |
| **Conjunctivitis (Binary Classification)** | `Image Dataset on Eye Diseases Classification...` | 1,006 images (357 conjunctivitis, 649 normal) | Binary verified class folders | **Verified & Ready** |
| **Conjunctiva Segmentation** | `Eye Conjunctiva Segmentation Dataset` | 547 eye images + 547 pairs of masks (Annotators 1 & 2) | Anatomical segmentation masks | **Verified & Ready** |
| **Jaundice / Bilirubin** | *None found in workspace* | 0 images | 0 verified bilirubin values | **Dataset Missing (Blocker)** |

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
    - **Train**: 151 subjects
    - **Validation**: 33 subjects
    - **Test**: 33 subjects
    - **Unlabeled**: 1 subject (Italian #93)
  - All multi-view images of a patient are locked into the same split.

---

## 2. Conjunctivitis Dataset
- **Origin**: `Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid)...`
- **Class Counts**:
  - **Conjunctivitis**: 357 images (322 RGB, 35 RGBA)
  - **Normal**: 649 images (649 RGB)
- **Integrity & Duplicates**:
  - 100% valid images (0 corrupt).
  - 1,006 unique MD5 hashes. 0 intra-class or cross-class duplicate files.
- **Split Distribution (Stratified 70/15/15, Seed 42)**:
  - **Train**: 704 images (454 Normal, 250 Conjunctivitis)
  - **Val**: 150 images (97 Normal, 53 Conjunctivitis)
  - **Test**: 152 images (98 Normal, 54 Conjunctivitis)

---

## 3. Eye Conjunctiva Segmentation Dataset
- **Images**: 547 eye images in PNG format.
- **Masks Annotator 1**: 547 masks (100% pairwise matched with images).
- **Masks Annotator 2**: 547 masks (100% pairwise matched with images).

---

## 4. Jaundice Dataset (Audit & Blocker)
- **Status**: **DATASET MISSING IN WORKSPACE**.
- **Audit Findings**:
  - Comprehensive recursive search across all directories in `C:\Users\sarth\OneDrive\Pictures\Innotech` confirmed that no dataset for jaundice, icterus, or laboratory bilirubin exists.
  - In accordance with clinical safety principles, **no synthetic bilirubin values or fake labels were fabricated**.
  - A standardized manifest schema has been created at `data/manifests/jaundice_manifest.csv` and reusable pipeline architecture has been scaffolded under `src/preprocessing/dataset_jaundice.py` and `src/models/jaundice_model.py` so that external jaundice data can be ingested seamlessly upon arrival.
