# ==============================================================================
# FuXi-S2S Model Service Dockerfile
# ==============================================================================
# GPU-enabled service for running FuXi-S2S model inference
# Port: 8002
# ==============================================================================

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ------------------------------------------------------------------------------
# Builder stage
# ------------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install numpy (compatible version)
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Install ONNX Runtime GPU
RUN pip install --no-cache-dir onnxruntime-gpu==1.16.3

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-multipart \
    httpx \
    pandas \
    xarray \
    netCDF4 \
    dask \
    bottleneck \
    pymongo \
    cdsapi \
    tqdm \
    coloredlogs \
    pyyaml \
    requests \
    openpyxl

# ------------------------------------------------------------------------------
# Runtime stage
# ------------------------------------------------------------------------------
FROM base AS runtime

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data/realtime /app/data/sample /app/output /app/model

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Environment variables
ENV PORT=8002
ENV MODEL_PATH=/app/model/fuxi_s2s.onnx
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/output
ENV DEVICE=cuda

# GPU settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
