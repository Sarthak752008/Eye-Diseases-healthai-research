# Jaundice Model Pipeline Status

## Operational Status: BLOCKED (DATA UNAVAILABLE)

### Blocker Details
- **Issue**: A thorough recursive audit of the workspace confirmed that **no dataset for jaundice, icterus, or laboratory bilirubin measurements exists**.
- **Clinical Safety Enforcement**: In strict compliance with `AGENTS.md` and clinical safety protocols, **no synthetic bilirubin values or fabricated labels have been created**.
- **Architecture Readiness**: The MobileNetV3-Small classifier architecture (`src/models/jaundice_model.py`) and sclera color-preserving preprocessing pipeline (`src/preprocessing/dataset_jaundice.py`) have been fully implemented, unit-tested, and verified.
- **Data Ingestion Readiness**: The standardized folder structure (`data/processed/jaundice/train/`, `val/`, `test/`) is established. Once an authorized clinical jaundice dataset is placed in `data/raw/jaundice/`, the pipeline will execute end-to-end immediately.
