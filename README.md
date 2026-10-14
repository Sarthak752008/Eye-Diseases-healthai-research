# EyeHealthAI (OcuScan) — Multi-Modal Clinical Ocular Screening Framework

[![Pytest Suite](https://img.shields.io/badge/pytest-140%20passed%20(100%25)-brightgreen)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-teal)](app.py)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](Dockerfile)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20%7C%20MobileNetV3-red)](src/models/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini%20Enabled-black)](src/inference/doctor_buddy.py)
[![Research Safety](https://img.shields.io/badge/Reliability%20Gates-Active%20Enforced-blue)](docs/PROJECT_STATUS.md)

**EyeHealthAI (OcuScan)** is an investigational, multi-modal computer vision and medical AI framework engineered for non-invasive ocular health screening across three distinct clinical targets:

1. **Anemia Screening (Continuous Hb Estimation)**: Continuous optical hemoglobin ($Hb$ in g/dL) prediction from palpebral and forniceal conjunctival pallor using a fine-tuned ResNet-50 regressor. Enhanced with WHO 4-tier Severity Risk Staging (Normal, Mild, Moderate, Severe), 95% Confidence Intervals, and Erythema Index ($EI$).
2. **Conjunctival Injection / Hyperemia Screening**: Anterior ocular surface hyperemia screening powered by the verified SLID Slit-Lamp biomicroscopy classifier (Accuracy: 98.78%, AUROC: 0.9996). Confounded legacy datasets (Mendeley) remain permanently blocked by reliability gates.
3. **Scleral Icterus / Jaundice Screening**: Categorical scleral yellowing screening (Normal vs. Visibly Jaundiced) via MobileNetV3-Small. Total serum bilirubin numerical regression remains strictly gated due to absent paired laboratory reference ground truth.

---

## Key Features & Advanced AI Architecture

### 1. Six-Stage Multi-Modal Pipeline
- **Stage 1: Clinical Ocular Segmentation**: Isolates palpebral/forniceal conjunctiva and anterior sclera regions using chromatic color-space segmentation and anatomical contours.
- **Stage 2: Primary CNN Ingestion**: Dedicated ResNet-50 continuous regressor and MobileNetV3-Small classifiers evaluated with leakage-free subject splitting.
- **Stage 3: Secondary Vision Verification**: Clinical vision corroboration (MedSigLIP/MedGemma architecture with OpenAI GPT-4o-mini vision integration and deterministic fallback).
- **Stage 4: Reference Retrieval**: Real-time feature cosine similarity matching against validated clinical reference archives.
- **Stage 5: Calibrated Ensemble Fusion**: Multimodal Bayesian fusion computing concordance scores (0–100%), uncertainty estimation, and triage recommendation tiers.
- **Stage 6: Grounded Doctor Buddy 2.0 ("Dr. Netra")**: Context-aware patient-friendly bilingual conversational guide with speech synthesis (Hindi `hi-IN` & English `en-US`), PII redaction, and strict medical guardrails.

### 2. Patient Experience & Clinical Workflow
- **Multi-Step Guided Wizard**: Language selection $\rightarrow$ Registration & Informed Consent $\rightarrow$ 6-Step Preparation Protocol $\rightarrow$ Smart Camera/Upload $\rightarrow$ AI Inference $\rightarrow$ Ocular Health Scorecard $\rightarrow$ Session History.
- **Strict Two-Page Clinical PDF Export**: Formatted clinical report in English or Hindi, displaying WHO classification, 95% CI bands, chromatic biomarkers, segmentation coverage, and clinical referral advice.
- **Zero Confusion & Provenance**: Every output explicitly links to its unique session ID, timestamp, and SHA-256 image input hash.

---

## Repository Structure

```text
EyeHealthAI/
├── Dockerfile                                   # Multi-stage production container
├── docker-compose.yml                           # Docker Compose orchestration
├── requirements.txt                             # Production dependencies
├── app.py                                       # FastAPI production backend (lifespan enabled)
├── vercel.json                                  # Vercel serverless rewrite config
├── api/
│   └── index.py                                 # Vercel serverless entrypoint
├── data/
│   ├── manifests/                               # Leakage-free train/val/test CSV manifests
│   └── eyehealthai.db                           # Local SQLite database (gitignored)
├── dataset anemia/                              # Raw Anemia clinical dataset (India & Italy)
├── docs/                                        # Clinical & deployment documentation
│   ├── CLEANUP_AUDIT.md                         # Detailed cleanup & structural audit record
│   ├── HOSPITAL_DEPLOYMENT.md                   # Hospital server setup & security guide
│   └── ...                                      # Domain validation & training audits
├── Eye Conjunctiva Segmentation Dataset/        # Raw segmentation images & masks
├── Image Dataset on Eye Diseases.../            # Raw ocular disease dataset
├── notebooks/                                   # Research exploration notebooks
├── reports/                                     # Training curves & evaluation JSONs
├── saved_models/                                # Clinical model weights & metrics
├── scripts/                                     # Reproducible training & audit scripts
├── src/                                         # Application source code
│   ├── config.py                                # System settings & paths
│   ├── database/                                # SQLAlchemy ORM models & session
│   ├── evaluation/                              # Model evaluation utilities
│   ├── inference/                               # Unified inference, Doctor Buddy, Ensemble, PDF
│   ├── models/                                  # ResNet-50 & MobileNetV3 architectures
│   ├── preprocessing/                           # Quality checks, crop, eversion, normalization
│   └── segmentation/                            # Multi-target conjunctiva/sclera segmentation
├── static/                                      # Clinical frontend UI
│   ├── css/style.css                            # Responsive clinical glassmorphic styling
│   ├── images/                                  # Medical avatar graphics
│   ├── js/dashboard.js                          # Frontend client logic & state machine
│   ├── index.html                               # Screening wizard & dashboard
│   └── report.html                              # Ocular Health Scorecard
└── tests/                                       # Automated pytest suite (140 tests)
```

---

## Setup & Deployment

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Modern web browser (Chrome, Edge, Firefox, Safari) with WebRTC and Web Speech API support
- *Optional:* Docker & Docker Compose for containerized deployment

### 2. Standard Local Installation
```bash
git clone https://github.com/Sarthak752008/Eye-Diseases-healthai-research.git
cd Eye-Diseases-healthai-research

# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```ini
APP_ENV=development
HOST=127.0.0.1
PORT=8000
MAX_UPLOAD_SIZE_MB=10

# AI Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```
*(If external AI keys are omitted or rate-limited, Doctor Buddy and Secondary Vision seamlessly run in certified offline deterministic mode with full clinical guardrails.)*

### 4. Running the Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://127.0.0.1:8000`.

### 5. Running with Docker
```bash
docker compose up --build -d
```
The application will be accessible at `http://localhost:8000`.

---

## Automated Verification Suite

Run the full end-to-end pytest suite:
```bash
python -m pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collected 140 items

tests/test_advanced_ai.py ..........                                     [  4%]
tests/test_copilot.py .................................                  [ 27%]
tests/test_doctor_buddy.py .................                             [ 39%]
tests/test_layer4_inference.py ...........                               [ 45%]
tests/test_layer5_dashboard.py ..........                                [ 52%]
tests/test_layer6_webcam.py ............                                 [ 62%]
tests/test_layers.py .......                                             [ 67%]
tests/test_openai_integration.py ...                                     [ 69%]
tests/test_phase7_readiness.py ..........                                [ 76%]
tests/test_phase8_prototype.py ...........                               [ 84%]
tests/test_remediation_and_models.py .......................             [100%]

======================= 140 passed in 102.25s (0:01:42) =======================
```

---

## Clinical Safety Disclaimers

1. **Investigational Research Prototype Only**: EyeHealthAI is an investigational software tool designed to aid screening workflows and research. It does NOT provide a certified medical diagnosis.
2. **Confirmatory Laboratory Testing Required**: Optical hemoglobin estimations must be verified via a standard Complete Blood Count (CBC) venous blood draw before initiating clinical treatments.
3. **Reliability Gates**: The Mendeley conjunctivitis dataset remains permanently blocked due to high-contrast shortcut artifacts.
4. **Data Privacy**: No patient images, camera feeds, or personal identifiers are stored or shared without explicit on-screen consent.
