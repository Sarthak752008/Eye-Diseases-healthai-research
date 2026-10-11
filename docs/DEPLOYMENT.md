# EyeHealthAI — Deployment & Operations Guide

**Application**: EyeHealthAI Multi-Modal Research Prototype  
**Framework**: FastAPI, Uvicorn, PyTorch, TorchVision  
**Target Environment**: Vercel Serverless / Linux / Windows / Cloud

---

## 1. Prerequisites & Resource Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Python** | 3.10+ | 3.11 |
| **System Memory (RAM)** | 2.5 GB | 4.0 GB+ |
| **Storage** | 1.0 GB (App + Dependencies) | 2.0 GB (with Model Checkpoints) |
| **Compute Device** | CPU | CPU or NVIDIA CUDA GPU (optional) |
| **Network** | HTTP / HTTPS port 8000 | HTTPS (mandatory for live browser webcam in production) |

---

## 2. Vercel Deployment

EyeHealthAI is configured for seamless deployment to **Vercel** via serverless Python entry points.

### Repository Files for Vercel:
- `vercel.json`: Directs all incoming HTTP requests to `/api/index.py`.
- `api/index.py`: Serverless ASGI entry point exposing the FastAPI application.
- `requirements.txt`: Python runtime dependencies.

### Steps to Deploy on Vercel:

1. **Push to GitHub**:
   Ensure the latest code is pushed to your GitHub repository:
   ```bash
   git push origin main
   ```

2. **Import into Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/new).
   - Connect your GitHub account and import `Sarthak752008/Eye-Diseases-healthai-research`.
   - Select **Other** as the Framework Preset.
   - Leave the Root Directory as `./`.

3. **Configure Environment Variables in Vercel**:
   In the Vercel project settings, add the following environment variables:
   - `ENV`: `production`
   - `ALLOWED_ORIGINS`: `https://your-vercel-app.vercel.app`
   - `DEVICE`: `cpu`
   - `MAX_UPLOAD_SIZE_MB`: `10`

4. **Click Deploy**:
   Vercel will install dependencies from `requirements.txt`, bundle `api/index.py`, and deploy the research dashboard.

> [!NOTE]
> **Vercel Serverless Architecture & Model Checkpoints**:
> - The application starts and runs in resilient mode without crashing if the 288 MB local binary weights (`best_model.pth`) are not present in the serverless bundle.
> - The health endpoint `/health` and `/api/health`, research dashboard UI, webcam capture interface, quality feedback, and gate systems operate normally.
> - For full local or dedicated server inference with heavy 288 MB PyTorch checkpoints, follow the Local Production Execution steps below.

---

## 3. Local Production Execution

### Step 1: Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the template configuration:
```bash
cp .env.example .env
```
Edit `.env` to suit your deployment parameters:
```ini
ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
MAX_UPLOAD_SIZE_MB=10
DEVICE=cpu
```

### Step 3: Verify Model Checkpoints
Ensure model checkpoints are placed in the expected directory structure:
```
saved_models/
├── anemia/
│   └── best_model.pth          # (Required for Anemia regression, ~288MB)
├── conjunctivitis/
│   └── best_model_v2_clean.pth # (Offline research artifact, ~12.5MB)
└── jaundice/
    └── jaundice_config.json    # (Architecture scaffold config)
```

### Step 4: Launch the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 4. Verification & Health Probes

The application provides dual health check endpoints:
- `GET /health` (Standard platform health probe)
- `GET /api/health` (Detailed API readiness probe)

### Verification:
```bash
curl -i http://127.0.0.1:8000/health
```

---

## 5. Troubleshooting & Operational Guidance

### A. Camera Permissions in Production
- **Symptom**: Browser reports `Camera Error: Permission Denied`.
- **Cause**: Modern web browsers require **HTTPS** (or `localhost`) to access `navigator.mediaDevices.getUserMedia()`.
- **Resolution**: Vercel provides automatic HTTPS certificates out of the box, ensuring the webcam interface works securely.

### B. Missing Checkpoint Error (`FileNotFoundError`)
- **Symptom**: Server starts, but Anemia analysis returns checkpoint not found.
- **Cause**: The application is resilient and starts even if optional weights are absent, but inference requires the actual weights.
- **Resolution**: Verify that `saved_models/anemia/best_model.pth` exists or configure `ANEMIA_CHECKPOINT_PATH`.
