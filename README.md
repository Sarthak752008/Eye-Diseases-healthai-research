# EyeHealthAI — Multi-Modal Ocular Health Screening Framework

[![Pytest Suite](https://img.shields.io/badge/pytest-123%20passed%20(100%25)-brightgreen)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-teal)](app.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20%7C%20MobileNetV3-red)](src/models/)
[![Research Safety](https://img.shields.io/badge/Reliability%20Gates-Active%20Enforced-blue)](docs/PROJECT_STATUS.md)

EyeHealthAI is an investigational computer-vision research framework for non-invasive ocular health screening across three distinct clinical targets:
1. **Anemia Screening**: Continuous hemoglobin ($Hb$ in g/dL) estimation from palpebral and forniceal conjunctival pallor using a fine-tuned ResNet-50 regressor, enhanced with WHO Severity Risk Staging, 95% Confidence Intervals, and Erythema Index ($EI$).
2. **Conjunctival Injection (Hyperemia)**: Anterior ocular surface inflammation screening powered by the verified SLID Slit-Lamp biomicroscopy classifier (Accuracy: 98.78%, AUROC: 0.9996). Legacy confounded datasets (Mendeley) remain permanently blocked by reliability gates.
3. **Scleral Icterus / Jaundice Screening**: Categorical yellowing screening (Normal vs. Visibly Jaundiced) via MobileNetV3-Small. Quantitative total bilirubin estimation remains strictly unavailable per clinical safety standards due to absent paired serum reference ground truth.

---

## Interactive Features & AI Doctor Buddy ("Dr. Netra")

### 1. AI Doctor Buddy Floating Mascot
- **Visual Design**: 2D animated cartoon doctor mascot (Dr. Netra) based on clean clinical aesthetic without watermarks.
- **Context-Aware Dialogue**: Real-time contextual guidance for permissions, macro positioning (10–15 cm), blur/glare alerts, full-face rejections, and borderline clinical zones.
- **Bilingual Web Speech (TTS)**: Built-in SpeechSynthesis with Hindi (`hi-IN`) and English (`en-US`) natural voice playback, voice toggle (`🔊/🔇`), replay, and accessible text fallback.
- **Interactive Expressions**: Context-switched avatar states (`default`, `speaking`, `alert`, `happy`).
- **Privacy-First Guardrails**: No raw images, camera frames, patient IDs, or medical data are ever transmitted to external cloud APIs.

### 2. Smart Eye-Only Capture & Framing
- **Anterior Ocular Region Isolation**: Live OpenCV eye-detector with real-time tracking reticle, rejecting distant full-face portraits.
- **Quality Pre-Screening**: Real-time Laplacian variance sharpness check and luminance thresholding (detecting blur, under-exposure, and flash reflections).
- **Graceful Error Handling**: Detects and guides denied camera permissions, missing devices, and unvalidated webcam domains.

### 3. Patient Journey & Reporting
- **Multi-Step Guided Wizard**: Language selection $\rightarrow$ Registration & Informed Consent $\rightarrow$ 6-Step Preparation $\rightarrow$ Guided Eye Capture $\rightarrow$ AI Inference $\rightarrow$ Scorecard $\rightarrow$ Session History.
- **Strict Two-Page PDF Clinical Export**: Fully formatted ReportLab clinical report in English or Hindi, displaying WHO classification, 95% CI, palpebral Erythema Index ($EI$), and safety boundaries.

---

## Architecture & Module Directory

```text
EyeHealthAI/
├── app.py                         # Unified FastAPI REST server & dashboard router
├── src/
│   ├── config.py                  # Central environment settings & GEMINI_MODEL configuration
│   ├── database/                  # SQLAlchemy schema, session lifecycle & GDPR purge
│   ├── inference/
│   │   ├── clinical_analytics.py  # WHO staging, 95% CI bands, and chromatic biomarkers
│   │   ├── copilot.py             # Gemini Research Copilot service with PII sanitization
│   │   ├── doctor_buddy.py        # Dr. Netra mascot knowledge base & safe LLM enrichment
│   │   ├── pdf_report.py          # 2-Page bilingual clinical PDF generation engine
│   │   ├── session_export.py      # Standardized session export schema validator
│   │   └── unified_service.py     # Multi-view aggregation & inference gateway
│   ├── models/                    # ResNet-50 and MobileNetV3 PyTorch architectures
│   └── preprocessing/             # Image quality, Haar cascade eye detector & view validation
├── static/
│   ├── css/style.css              # Dark clinical glassmorphic styling & mascot animations
│   ├── js/dashboard.js            # Frontend journey logic, camera stream & voice synthesis
│   ├── images/                    # Doctor Buddy avatar expressions & mascot assets
│   └── index.html                 # Complete responsive 7-step screening application
├── tests/                         # Complete Pytest automated verification suite (123 tests)
└── docs/                          # Clinical audits, safety reports, and model comparisons
```

---

## Setup & Local Deployment

### 1. Requirements
- Python 3.10+ (tested on Python 3.11)
- Modern web browser (Chrome, Edge, Firefox, Safari) with WebRTC and Web Speech API support

### 2. Installation
```bash
git clone https://github.com/Sarthak752008/Eye-Diseases-healthai-research.git
cd Eye-Diseases-healthai-research
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```ini
ENV=development
DEBUG=false
HOST=127.0.0.1
PORT=8000
ALLOWED_ORIGINS=*
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```
*(If `GEMINI_API_KEY` is omitted or quota is exceeded, Doctor Buddy and Research Copilot automatically operate in deterministic offline mode.)*

### 4. Running the Server
```bash
python app.py
```
Open your browser at `http://127.0.0.1:8000`.

### 5. Running the Automated Tests
```bash
python -m pytest tests/ -v
```
All **123 automated tests** should pass with 100% success rate.

---

## Clinical Safety Disclaimers
1. **Research Prototype Only**: EyeHealthAI is an investigational software prototype and does not provide a certified medical diagnosis.
2. **Confirmatory Laboratory Testing Required**: Continuous optical hemoglobin estimations require verification through a standardized Complete Blood Count (CBC) venous blood draw.
3. **No Shortcut Bypasses**: The Mendeley conjunctivitis model remains blocked due to a 100% resolution/framing shortcut in the source dataset.
