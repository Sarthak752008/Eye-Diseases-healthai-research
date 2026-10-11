# EyeHealthAI — Deployment & Operations Guide

**Application**: EyeHealthAI Multi-Modal Research Prototype  
**Framework**: FastAPI, Uvicorn, PyTorch, TorchVision  
**Target Environment**: Linux / Windows / Docker Container (x86_64 or ARM64)

---

## 1. Prerequisites & Resource Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Python** | 3.10+ | 3.11 |
| **System Memory (RAM)** | 2.5 GB | 4.0 GB+ |
| **Storage** | 1.0 GB (App + Dependencies) | 2.0 GB (with Model Checkpoints) |
| **Compute Device** | CPU | CPU or NVIDIA CUDA GPU (optional) |
| **Network** | HTTP / HTTPS port 8000 | HTTPS (mandatory for live browser webcam in production) |

> [!WARNING]
> **Model Checkpoint Size & Memory Consideration**:
> The primary Anemia ResNet-50 PyTorch checkpoint (`best_model.pth`) is **~288 MB**. Loading this model into memory consumes approximately 600–800 MB of RAM per process worker. Ensure your hosting target or container has at least **2.5 GB of free RAM** to prevent out-of-memory (OOM) process termination.

---

## 2. Quickstart (Local Production Execution)

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
ALLOWED_ORIGINS=https://your-domain.org
MAX_UPLOAD_SIZE_MB=10
DEVICE=cpu
```

### Step 3: Verify Model Checkpoints
Ensure the model checkpoints are placed in the expected directory structure:
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
# Recommended production start command:
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

> [!NOTE]
> For single-node CPU deployments, `--workers 1` or `--workers 2` is recommended to avoid redundant multi-process duplication of the 288 MB ResNet-50 weights in RAM.

---

## 3. Docker Deployment

### Step 1: Build Container Image
```bash
docker build -t eyehealthai:latest .
```

### Step 2: Run Container with Mounted Checkpoint Volume
Because large binary checkpoints (`*.pth`) are excluded from version control to protect repository limits, mount your local `saved_models` directory at runtime:

```bash
docker run -d \
  --name eyehealthai-app \
  -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models:ro \
  -e ENV=production \
  -e ALLOWED_ORIGINS="http://localhost:8000" \
  eyehealthai:latest
```

### Step 3: Verify Container Health
```bash
docker ps
# Inspect health status:
docker inspect --format='{{json .State.Health.Status}}' eyehealthai-app
```

---

## 4. Verification & Health Probes

The application provides dual health check endpoints suitable for load balancers, Kubernetes probes, and monitoring tools:

- `GET /health` (Standard container health probe)
- `GET /api/health` (API health probe)

### Verification Commands:
```bash
# Using curl:
curl -i http://127.0.0.1:8000/health

# Using PowerShell:
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

### Sample Health Output:
```json
{
  "status": "healthy",
  "service": "EyeHealthAI Research Service",
  "version": "1.0.0",
  "environment": "production",
  "device": "cpu",
  "max_upload_size_mb": 10,
  "modules": {
    "anemia": {
      "checkpoint_present": true,
      "operational_status": "available_research_only",
      "validation_status": "experimental_research_only",
      "notes": "Non-diagnostic research regressor. Paired venous reference testing required."
    },
    "conjunctivitis": {
      "checkpoint_present": true,
      "operational_status": "blocked_validation",
      "validation_status": "shortcut_confound_blocked",
      "notes": "Predictions withheld due to 100% resolution/framing dataset shortcut."
    },
    "jaundice": {
      "checkpoint_present": false,
      "operational_status": "unavailable_missing_data",
      "validation_status": "unimplemented_no_ground_truth",
      "notes": "No clinical dataset or laboratory bilirubin reference measurements exist."
    }
  },
  "disclaimer": "RESEARCH USE ONLY: Prototype service. Not certified or validated for clinical diagnostic use."
}
```

---

## 5. Troubleshooting & Operational Guidance

### A. Camera Permissions in Production
- **Symptom**: Browser reports `Camera Error: Permission Denied` or camera will not start.
- **Cause**: Modern web browsers require a **Secure Context (HTTPS or localhost)** to access `navigator.mediaDevices.getUserMedia()`.
- **Resolution**: Deploy behind a TLS/SSL reverse proxy (e.g., Nginx, Caddy, Cloudflare) with valid HTTPS certificates.

### B. Missing Checkpoint Error (`FileNotFoundError`)
- **Symptom**: Server starts, but Anemia analysis returns `Anemia checkpoint not found at: saved_models/anemia/best_model.pth`.
- **Cause**: The application is resilient and starts without crashing even if optional weights are absent, but inference requires the actual weights.
- **Resolution**: Verify that `saved_models/anemia/best_model.pth` exists or set the `ANEMIA_CHECKPOINT_PATH` environment variable to its location.

### C. Large Upload Rejection (`HTTP 413`)
- **Symptom**: Upload fails with `File exceeds maximum allowed upload size`.
- **Resolution**: Increase `MAX_UPLOAD_SIZE_MB` in `.env` (default is 10 MB) if high-resolution uncompressed ocular photography is being tested.

### D. CORS Blocking (`CORS header 'Access-Control-Allow-Origin' missing`)
- **Symptom**: Frontend cannot communicate with API on a custom domain.
- **Resolution**: Update `ALLOWED_ORIGINS` in `.env` to include your exact frontend domain and scheme (e.g. `ALLOWED_ORIGINS=https://eyehealth.example.org`).
