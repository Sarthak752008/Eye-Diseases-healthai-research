# AGENTS.md — EyeHealthAI Project Architecture & Standards

## Project Overview
EyeHealthAI is a multi-modal computer vision framework for non-invasive ocular health screening across three distinct clinical targets:
1. **Layer 1: Dataset Audit & Manifest Integrity** — Verifying genuine clinical laboratory measurements (Hb, Bilirubin), paired masks, and resolving patient/session-level structures.
2. **Layer 2: Preprocessing & Leakage-Free Splitting** — Safe recursive image ingestion, corrupt-file protection, blur/illumination quality verification, color-space preservation (no aggressive color jitter that distorts conjunctival pallor or scleral icterus), and patient/session group-aware train/val/test splits.
3. **Layer 3: Three Independent Specialized Models**:
   - **Model A (Anemia / Hb Regression)**: Pretrained ResNet-50 / CNN regressor predicting laboratory hemoglobin (Hb in g/dL). Evaluated using MAE and RMSE.
   - **Model B (Conjunctivitis Classification)**: Pretrained MobileNetV3-Small binary classifier (Normal vs. Conjunctivitis).
   - **Model C (Jaundice Classifier)**: Pretrained MobileNetV3-Small binary classifier (Normal vs. Visibly Jaundiced).

## Clinical Safety & Ground-Truth Rules
1. **Never fabricate ground truth**: Numerical laboratory Hb and bilirubin values must strictly originate from clinical laboratory reference spreadsheets (`India.xlsx`, `Italy.xlsx`). If labels are unavailable (e.g. Italian patient #93 or missing Jaundice dataset), explicitly report the limitation.
2. **Preserve clinical colors**: Do not apply aggressive hue/saturation shifts, solarization, or color inversions. Conjunctival pallor (erythema reduction) and scleral icterus (bilirubin staining) are chromatic signals.
3. **Prevent data leakage**: All views/sessions belonging to the same subject must strictly reside in the same partition (train, val, or test).
4. **Independent model heads**: Never merge the three clinical tasks into a single monolithic classifier. Each disease domain requires separate loss functions and evaluation metrics.
5. **Traceability**: All processed images and manifests must retain clear traceability back to raw filesystem paths.
