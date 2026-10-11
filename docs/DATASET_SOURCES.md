# Dataset Sources & Licensing Audit

**Document Date**: 2026-10-11  
**Auditor**: Senior ML Engineer & CV Researcher  
**Scope**: Verification of data sources, licensing terms, clinical ground-truth availability, and acquisition status across all three clinical targets.

---

## 1. Summary of Evaluated Sources

| Target | Source Name / Repository | Access URL / DOI | Licence | Ground Truth Type | Technical Status | Deployment Permissibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Anemia** | **Eyes-defy-anemia** (India & Italy cohorts) | Local clinical reference spreadsheets (`India.xlsx`, `Italy.xlsx`) | Academic Research Use | Genuine Venous Blood Laboratory Hemoglobin ($g/dL$) | **Verified & Ready** (862 images, 218 subjects) | Research screening only; patient-grouped partitions |
| **Model B: Conjunctival Injection** | **SLID (Slit-Lamp Image Dataset)** | `https://github.com/xumingyu-hub/SLID`<br>`https://doi.org/10.3389/fdgth.2025.1716501` | CC BY 4.0 (Open Access) | Clinical ophthalmological lesion annotation: `Conjunctival injection` ($N=307$) vs Zero Lesions / Healthy ($N=245$) | **Acquired & Verified** (2,617 clinical slit-lamp images, $1924\times 1556$ / $2576\times 1934$ px) | Fully permitted for academic & research use; matched high-resolution slit-lamp optics |
| **Model B: Confounded Source** | **Mendeley Data v2** (`n9zp473wfw/2`) | `https://data.mendeley.com/datasets/n9zp473wfw/2` | CC BY 4.0 | Web-scraped disease categories | **Audited & Gated** (100% resolution confounding: Normal 56 px vs Conjunctivitis 352 px) | Blocked by reliability gate in clinical inference |
| **Model C: Jaundice** | **Bleachcarte Jaundice Dataset** | `https://huggingface.co/datasets/Bleachcarte/Jaundice_Dataset` | **CC BY-NC 4.0** | Web-scraped categorical labels: `Healthy` ($N=176$) vs `Jaundice` ($N=145$) | **Evaluated & Verified** (321 total images across train & test splits) | **Non-Commercial Research Only**. Strictly blocked from numerical bilirubin estimation. |
| **Model C: Jaundice (Clinical)** | **UCL Research Data Repository** | `https://doi.org/10.5522/04/24083487.v1` | CC0 1.0 (Data record) | Laboratory serum bilirubin in cirrhosis patients | **Images Restricted** (Repository distributes processed Bland-Altman spreadsheets; raw patient facial photographs held under NHS IRB 08/H0714/8) | Numerical lab ground truth exists in tabular form, but raw ocular pixels are restricted |

---

## 2. In-Depth Inspection Findings

### 2.1 Model A: Anemia Dataset (India & Italy)
- **Origin**: Clinical calibrated LED anterior eye photography from Bari, Italy and Chennai, India.
- **Verification**: Complete 1-to-1 traceability back to original Excel spreadsheets:
  - `India.xlsx`: 95 patients with venous blood Hb measurements (7.0–17.4 g/dL).
  - `Italy.xlsx`: 123 patients with venous blood Hb measurements (Subject #93 has missing `_` and is properly marked unlabeled).
- **Leakage Prevention**: Subject-level holdout split (Train: 151 subjects, Val: 33 subjects, Test: 33 subjects). All multi-view crops of a single patient reside strictly in one partition.

### 2.2 Model B: SLID Dataset (Clinical Slit-Lamp Photography)
- **Origin**: Mingyu Xu et al., Eye Center of the Second Affiliated Hospital, Zhejiang University School of Medicine, published in *Frontiers in Digital Health* (2025).
- **Annotation Structure**:
  - `Annotations.csv`: 10,204 annotations across 2,617 images.
  - Ocular anatomical segmentations: `Pupil` (circle/ellipse), `Cornea` (circle/ellipse), `Conjunctiva` (polygon).
  - Lesion categories: 13 distinct categories including `Conjunctival injection` ($N=307$), `Keratitis` ($N=222$), `Pinguecula` ($N=405$), `Pigmented nevus` ($N=416$), etc.
  - Zero-lesion healthy anterior segment controls: $N=245$.
- **Optical Resolution & Framing**: All images are captured on digital slit lamps at $1924 \times 1556$ or $2576 \times 1934$ resolution. Normal controls and conjunctival injection cases have identical high spatial resolution, solving the Nyquist-frequency shortcut found in the Mendeley dataset.
- **Clinical Scope & Boundaries**: The dataset supports detection of **visible conjunctival injection (hyperemia)**. It does NOT identify microbiological etiology (bacterial, viral, chlamydial) or allergic triggers. In accordance with clinical safety principles, any model trained on this data is explicitly designated as a **Visible Conjunctival Injection / Hyperemia Research Model**, not a definitive conjunctivitis etiology test.

### 2.3 Model C: Jaundice Candidates
1. **Hugging Face (`Bleachcarte/Jaundice_Dataset`)**:
   - Curated by Daniel Ekong with explicit attention to skin tone diversity (including specific queries for jaundice across Fitzpatrick phototypes).
   - 321 web images (Train: 256 [141 Healthy, 115 Jaundice]; Test: 65 [35 Healthy, 30 Jaundice]).
   - **Licensing Constraint**: CC BY-NC 4.0 forbids commercial use without separate agreement.
   - **Clinical Constraint**: Purely categorical (healthy vs visibly jaundiced sclera). Does not contain total serum bilirubin (mg/dL). Cannot be used for quantitative bilirubin regression.
2. **UCL Cirrhosis Dataset (`10.5522/04/24083487.v1`)**:
   - Demonstrates smartphone sclera and eyelid colorimetry against laboratory bilirubin in cirrhosis patients.
   - However, the public deposit contains only the downstream Bland-Altman regression tables. Raw patient images cannot be redistributed publicly under institutional ethics constraints.

---

## 3. Data Ingestion & Manifest Roadmap

1. **Anemia Manifest**: Retained at `data/manifests/anemia_manifest.csv` with 862 records across 218 subjects.
2. **SLID Manifest**: Scaffolded extraction of the 307 conjunctival injection and 245 normal control images with patient-grouped / source-isolated splits into `data/manifests/slid_injection_manifest.csv`.
3. **Jaundice Manifest**: Created from `Bleachcarte/Jaundice_Dataset` with honest categorical target (`is_visibly_jaundiced: 0 or 1`) and explicit CC BY-NC 4.0 attribution in `data/manifests/jaundice_manifest.csv`.
