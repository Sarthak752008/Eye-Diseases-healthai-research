# EyeHealthAI (OcuScan) — Multi-Modal Clinical Ocular Screening Framework

[![Pytest Suite](https://img.shields.io/badge/pytest-200%2B%20passed%20(100%25)-brightgreen)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-teal)](app.py)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](Dockerfile)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20%7C%20MobileNetV3-red)](src/models/)
[![Grad-CAM](https://img.shields.io/badge/Grad--CAM-Attribution%20Heatmaps-purple)](src/inference/attribution_cam.py)
[![Research Safety](https://img.shields.io/badge/Reliability%20Gates-Active%20Enforced-blue)](docs/COMPLETE_EVALUATION_AUDIT.md)

**EyeHealthAI (OcuScan)** is an investigational, multi-modal computer vision and medical AI framework engineered for non-invasive ocular health screening across three distinct clinical targets:

1. **Anemia Screening (Continuous Hb Estimation)**: Continuous optical hemoglobin ($Hb$ in g/dL) prediction from palpebral and forniceal conjunctival pallor using a fine-tuned ResNet-50 regressor. Evaluated using MAE and RMSE against laboratory venous blood references (`India.xlsx`, `Italy.xlsx`). Enhanced with demographic age-stratified WHO 4-tier Severity Risk Staging (Normal, Mild, Moderate, Severe), 95% Confidence Intervals, and Erythema Index ($EI$).
2. **Conjunctival Injection / Hyperemia Screening**: Anterior ocular surface hyperemia screening powered by the verified SLID Slit-Lamp biomicroscopy classifier (Accuracy: 98.78%, AUROC: 0.9996). Confounded legacy datasets (Mendeley) remain permanently blocked by reliability gates.
3. **Scleral Icterus / Jaundice Screening**: Categorical scleral yellowing screening (Normal vs. Visibly Jaundiced) via MobileNetV3-Small. Total serum bilirubin numerical regression remains strictly gated due to absent paired laboratory reference ground truth.

---

## 🔬 Core Medical AI Architecture & Pipeline

### 1. Three-Stage Transparent Visual Pipeline
To ensure complete clinical transparency and provenance from raw ingestion to model attribution, the system implements a verified 3-stage visual sequence across the web gallery, workspace, and clinical PDF reports:

- **01 Original Upload**: High-resolution unaltered ocular photograph ingested directly from the camera or file upload.
- **02 Final Model Input**: Exact 224×224 normalized RGB tensor fed into the deep convolutional networks.
- **03 AI Attention Overlay**: Genuine Grad-CAM feature attribution heatmap highlighting the anatomical regions that directly drove the prediction (e.g. palpebral conjunctival microvascular bed for Anemia, anterior vessel engorgement for SLID, or scleral pigmentation for Jaundice).

### 2. Anatomical Preservation & Macro-Eye Detection
- **Direct Macro Ocular Recognition**: Automated detection of close-up eye photographs and everted lower eyelids. Unlike generic face detectors that hallucinate partial face meshes or apply destructive center crops, EyeHealthAI preserves 100% of the visible palpebral conjunctiva, eyelid margin, and sclera.
- **MediaPipe Face Landmarker & OpenCV Cascades**: When full-face portraits are presented, anatomical landmark geometry extracts the primary ocular region with generous horizontal and vertical orbital margins.
- **Conservative Color Enhancement**: Illumination correction strictly performed on the Luminance ($L$) channel of CIE-LAB color space, preserving chromatic $A$ and $B$ channels to prevent artificial discoloration of conjunctival erythema or scleral icterus.

### 3. Profile-Aware Clinical Intelligence & Demographic Stratification
- **Age-Stratified Reference Ranges**: WHO clinical hemoglobin cutoffs adapt dynamically to patient age and clinical sex (Pediatric $<12\text{y}$, Adult Female, Adult Male, Geriatric $>65\text{y}$).
- **Multi-Modal Calibrated Ensemble**: Bayesian concordance fusion combining CNN regression/classification, chromatic colorimetry ($EI$ and $B/G$ ratios), and secondary vision verification.

### 4. Interactive Web Scorecard & Two-Page Clinical PDF
- **Bilingual Interface**: Seamless instant toggling between English (`en-US`) and Hindi (`hi-IN`).
- **Interactive Attention Explorer**: Live switching between Anemia (ResNet-50 layer4), SLID (MobileNetV3 features), and Jaundice Grad-CAM overlays.
- **Strict Two-Page PDF Report**: Standardized, patient-friendly screening summaries and technical research scorecards conforming strictly to a 2-page print layout.

---

## 📁 Repository Structure

```text
EyeHealthAI/
├── Dockerfile                                   # Multi-stage production container
├── docker-compose.yml                           # Docker Compose orchestration
├── requirements.txt                             # Production dependencies
├── app.py                                       # FastAPI production backend (lifespan enabled)
├── checkpoints/                                 # MediaPipe & OpenCV landmark detectors
│   ├── blaze_face_short_range.tflite
│   └── face_landmarker.task
├── data/
│   ├── manifests/                               # Leakage-free train/val/test CSV manifests
│   └── eyehealthai.db                           # Local SQLite database (gitignored)
├── dataset anemia/                              # Raw Anemia clinical dataset (India & Italy)
├── docs/                                        # Clinical & deployment documentation
│   ├── COMPLETE_EVALUATION_AUDIT.md             # Complete end-to-end evaluation & audit
│   ├── DEMOGRAPHIC_EVALUATION_REPORT.md         # Demographic & age-stratified performance
│   └── HOSPITAL_DEPLOYMENT.md                   # Hospital server setup & security guide
├── saved_models/                                # Clinical model weights & metrics
│   ├── anemia/best_model.pth                    # Pretrained ResNet-50 Anemia regressor
│   ├── conjunctivitis/slid_injection_model.pth  # Pretrained MobileNetV3 SLID classifier
│   └── jaundice/best_model.pth                  # Pretrained MobileNetV3 Jaundice classifier
├── scripts/                                     # Training, evaluation & migration scripts
│   └── migrate_db_age_profile.py               # SQLite schema migration for patient profiles
├── src/                                         # Application source code
│   ├── config.py                                # System settings & paths
│   ├── database/                                # SQLAlchemy ORM models, session & schema
│   ├── evaluation/                              # Model evaluation & demographic metrics
│   ├── inference/                               # Unified inference, attribution & PDF generation
│   │   ├── attribution_cam.py                   # High-resolution Grad-CAM engine
│   │   ├── calibrated_ensemble.py               # Multi-modal Bayesian concordance fusion
│   │   ├── pdf_report.py                        # Strict 2-page ReportLab PDF generator
│   │   ├── personalized_ai.py                   # Demographic profile-aware clinical analysis
│   │   └── unified_service.py                   # 3-stage unified inference service
│   ├── models/                                  # ResNet-50 & MobileNetV3 architectures
│   ├── preprocessing/                           # Image quality, color-safe enhancements & crops
│   │   ├── advanced_pipeline.py                 # MediaPipe & macro ocular detection pipeline
│   │   └── image_quality.py                     # Laplacian sharpness & exposure validation
│   └── segmentation/                            # Multi-target conjunctiva/sclera segmentation
├── static/                                      # Clinical frontend UI
│   ├── css/style.css                            # Responsive clinical glassmorphic styling
│   ├── js/dashboard.js                          # Client state machine, camera & 3-stage preview
│   ├── index.html                               # Screening wizard & dashboard
│   └── report.html                              # Interactive Ocular Health Scorecard
└── tests/                                       # Comprehensive automated pytest suite (200+ tests)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Modern web browser (Chrome, Edge, Firefox, Safari) with WebRTC support
- Optional: Docker & Docker Compose

### 2. Standard Local Installation
```bash
# Clone the repository
git clone https://github.com/Sarthak752008/Eye-Diseases-healthai-research.git
cd Eye-Diseases-healthai-research

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
```bash
python app.py
```
Or with Uvicorn:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000`.

### 4. Running with Docker
```bash
docker compose up --build -d
```
The application will be accessible at `http://localhost:8000`.

---

## 🧪 Automated Verification Suite

Run the full end-to-end pytest suite across models, manifests, demographic profiling, and inference:
```bash
pytest tests/ -v
```

All 22 core remediation and model test cases verify:
```bash
pytest tests/test_remediation_and_models.py -v
```
- Strict zero subject leakage across train/val/test splits.
- Preprocessing contract identicality between training and inference.
- Deterministic repeated inference on real ocular photographs.
- 100% adherence to 3-stage visual sequence and 2-page PDF formatting.

---

## 🛡️ Clinical Safety & Ethics Rules

1. **Investigational Research Prototype Only**: EyeHealthAI is an investigational software tool designed to aid screening workflows and research. It does NOT provide a certified medical diagnosis.
2. **Confirmatory Laboratory Testing Required**: Optical hemoglobin estimations must be verified via a standard Complete Blood Count (CBC) venous blood draw before initiating clinical treatments.
3. **No Fabricated Ground Truth**: Numerical laboratory measurements (Hb and bilirubin) strictly originate from certified clinical laboratory reference spreadsheets (`India.xlsx`, `Italy.xlsx`). If labels are unavailable, limitations are transparently reported.
4. **Reliability Gates**: The Mendeley conjunctivitis dataset remains permanently blocked due to high-contrast shortcut artifacts.
5. **Data Privacy**: No patient images, camera feeds, or personal identifiers are stored or shared without explicit on-screen consent.
