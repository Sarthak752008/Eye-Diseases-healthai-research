# ==============================================================================
# EyeHealthAI — Production Dockerfile
# Multi-modal ocular computer vision research prototype
# ==============================================================================

FROM python:3.11-slim

# Set environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cpu \
    MAX_UPLOAD_SIZE_MB=10

WORKDIR /app

# Install minimal OS dependencies for image processing and health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root system user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application source code and assets
COPY app.py .
COPY src/ ./src/
COPY static/ ./static/
COPY docs/ ./docs/
COPY AGENTS.md .
COPY .env.example .

# Create saved_models structure
RUN mkdir -p /app/saved_models/anemia \
             /app/saved_models/conjunctivitis \
             /app/saved_models/jaundice && \
    chown -R appuser:appuser /app

# Switch to non-privileged user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server entry point
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
