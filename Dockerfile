# =============================================================================
# Dockerfile — Skin Lesion Classification API
# =============================================================================
#
# CPU-only PyTorch build. No CUDA required.
#
# Build:
#   docker build -t skin-cancer-backend .
#
# Run:
#   docker run -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -e MODEL_CHECKPOINT=models/skin_lesion_resnet18.pth \
#     -e ALLOWED_ORIGINS=http://localhost:3000 \
#     skin-cancer-backend
#
# =============================================================================

FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
# Install CPU-only PyTorch first using the official CPU index to avoid
# pulling the CUDA variant (which would add ~3 GB to the image size).
RUN pip install --no-cache-dir \
    torch==2.14.0 \
    torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt \
    # torch and torchvision are already installed above — skip re-download
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY app/ ./app/

# Models directory — mount a volume here in production to inject the checkpoint
# The API gracefully handles a missing checkpoint (starts but /predict returns 503)
RUN mkdir -p models

# ---------------------------------------------------------------------------
# Runtime configuration defaults (override via environment variables)
# ---------------------------------------------------------------------------
ENV APP_ENV=production
ENV MODEL_TYPE=resnet18
ENV MODEL_CHECKPOINT=models/skin_lesion_resnet18.pth
ENV ALLOWED_ORIGINS=http://localhost:3000
ENV MAX_UPLOAD_SIZE_MB=10

# Expose the API port
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check — Docker will call this to verify the container is alive
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------------------------
# Production server
# ---------------------------------------------------------------------------
# workers=1: PyTorch models are not designed for multi-process shared memory.
# For horizontal scaling, run multiple containers behind a load balancer.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
