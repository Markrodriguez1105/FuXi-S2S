# ==============================================================================
# FuXi-S2S Weather Forecast Model - Docker Image
# ==============================================================================
# Multi-stage build for optimized container size
# Supports CUDA GPU acceleration for ONNX Runtime inference
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image with CUDA support
# ------------------------------------------------------------------------------
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
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
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Set Python to not buffer output (for real-time logging)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ------------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# ------------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.7 support
RUN pip install --no-cache-dir \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install numpy first with compatible version (numpy 2.x breaks onnxruntime)
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Install ONNX Runtime GPU (CUDA 11.x compatible)
RUN pip install --no-cache-dir onnxruntime-gpu==1.16.3

# Install remaining requirements (excluding torch packages which are already installed)
RUN pip install --no-cache-dir \
    xarray \
    dask \
    netCDF4 \
    bottleneck \
    pandas \
    matplotlib \
    cdsapi \
    pymongo \
    tqdm \
    pyyaml \
    requests \
    click \
    coloredlogs \
    openpyxl

# ------------------------------------------------------------------------------
# Stage 3: Runtime - Final production image
# ------------------------------------------------------------------------------
FROM base AS runtime

# Set working directory
WORKDIR /app/fuxi-s2s

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/fuxi-s2s/data/realtime \
             /app/fuxi-s2s/data/sample \
             /app/fuxi-s2s/output \
             /app/fuxi-s2s/model \
             /root/.cdsapirc_dir

# Copy entrypoint script and make executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Environment variables for CDS API (can be overridden at runtime)
ENV CDS_API_URL=""
ENV CDS_API_KEY=""

# Environment variables for MongoDB (can be overridden at runtime)
ENV MONGO_DB_URI="mongodb://mongodb:27017"
ENV MONGO_URI="mongodb://mongodb:27017"
ENV MONGO_DB="arice"

# GPU settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port for potential web interface (future use)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import onnxruntime; print('OK')" || exit 1

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["--help"]
