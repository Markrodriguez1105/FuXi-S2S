# FuXi-S2S Docker Deployment Guide

This guide explains how to run the FuXi-S2S weather forecast model using Docker containers with GPU acceleration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Building the Image](#building-the-image)
4. [Running the Container](#running-the-container)
5. [Configuration](#configuration)
6. [Pipeline Commands](#pipeline-commands)
7. [Data Management](#data-management)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA 11.7+ support (recommended: RTX 3060 or better)
- 16GB+ RAM
- 50GB+ free disk space

### Software Requirements

- [Docker](https://docs.docker.com/get-docker/) (20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Install NVIDIA Container Toolkit (Linux)

```bash
# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Install NVIDIA Container Toolkit (Windows with WSL2)

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Enable WSL2 backend in Docker Desktop settings
3. GPU support is automatic with the latest Docker Desktop

---

## Quick Start

### 1. Clone and Setup

```bash
cd /path/to/FuXi-S2S
```

### 2. Configure CDS API Credentials

Create a `.cdsapirc` file in the project root:

```bash
# .cdsapirc
url: https://cds.climate.copernicus.eu/api
key: YOUR_CDS_API_KEY
```

Or set environment variables in `.env`:

```bash
# .env
CDS_API_URL=https://cds.climate.copernicus.eu/api
CDS_API_KEY=YOUR_CDS_API_KEY
```

### 3. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start MongoDB
docker-compose up -d mongodb

# Run the full forecast pipeline
docker-compose run --rm fuxi-s2s pipeline
```

---

## Building the Image

### Standard Build

```bash
docker-compose build fuxi-s2s
```

### Build with No Cache (fresh build)

```bash
docker-compose build --no-cache fuxi-s2s
```

### Build Only (without compose)

```bash
docker build -t fuxi-s2s:latest .
```

---

## Running the Container

### Using Docker Compose (Recommended)

```bash
# View available commands
docker-compose run --rm fuxi-s2s --help

# Run full pipeline with auto date detection
docker-compose run --rm fuxi-s2s pipeline

# Run pipeline for specific date
docker-compose run --rm fuxi-s2s pipeline --date 20250109

# Run with custom parameters
docker-compose run --rm fuxi-s2s pipeline --date 20250109 --members 11 --steps 42

# Interactive shell
docker-compose run --rm fuxi-s2s shell

# Run Python interpreter
docker-compose run --rm fuxi-s2s python
```

### Using Docker Directly

```bash
# Run with GPU access
docker run --gpus all -it --rm \
    -v $(pwd)/model:/app/fuxi-s2s/model:ro \
    -v $(pwd)/data:/app/fuxi-s2s/data \
    -v $(pwd)/output:/app/fuxi-s2s/output \
    -v $(pwd)/.cdsapirc:/root/.cdsapirc:ro \
    -e MONGO_URI=mongodb://host.docker.internal:27017 \
    fuxi-s2s:latest pipeline
```

---

## Configuration

### Environment Variables

| Variable               | Description               | Default                                 |
| ---------------------- | ------------------------- | --------------------------------------- |
| `CDS_API_URL`          | CDS API endpoint URL      | `https://cds.climate.copernicus.eu/api` |
| `CDS_API_KEY`          | Your CDS API key          | (required for download)                 |
| `MONGO_URI`            | MongoDB connection string | `mongodb://mongodb:27017`               |
| `MONGO_DB`             | MongoDB database name     | `arice`                                 |
| `CUDA_VISIBLE_DEVICES` | GPU device to use         | `0`                                     |

### Volume Mounts

| Host Path     | Container Path          | Purpose                   |
| ------------- | ----------------------- | ------------------------- |
| `./model`     | `/app/fuxi-s2s/model`   | ONNX model files          |
| `./data`      | `/app/fuxi-s2s/data`    | Input data (ERA5, sample) |
| `./output`    | `/app/fuxi-s2s/output`  | Forecast outputs          |
| `./.cdsapirc` | `/root/.cdsapirc`       | CDS API credentials       |
| `./compare`   | `/app/fuxi-s2s/compare` | Comparison results        |

---

## Pipeline Commands

### Full Pipeline

Runs the complete forecast workflow:

1. Download ERA5 data
2. Run FuXi-S2S inference
3. Store results to MongoDB
4. Verify storage

```bash
docker-compose run --rm fuxi-s2s pipeline [OPTIONS]
```

**Options:**

- `--date YYYYMMDD` - Initialization date (default: auto-detect latest available)
- `--members N` - Number of ensemble members (default: 11)
- `--steps N` - Forecast steps/days (default: 42)
- `--station NAME` - Station name for MongoDB storage

### Download Only

Download ERA5 data without running inference:

```bash
docker-compose run --rm fuxi-s2s download --date 20250109
```

### Inference Only

Run model inference on existing data:

```bash
docker-compose run --rm fuxi-s2s inference --members 11 --steps 42
```

### Store to MongoDB

Store existing forecasts to MongoDB:

```bash
docker-compose run --rm fuxi-s2s store --date 20250109 --members 11
```

### Verify Storage

Check MongoDB storage:

```bash
docker-compose run --rm fuxi-s2s verify
```

---

## Data Management

### Download Model Files

The ONNX model file is not included in the Docker image. Download it from Zenodo:

```bash
# Download model (approximately 1.2 GB)
wget https://zenodo.org/records/15718402/files/fuxi_s2s.onnx -O model/fuxi_s2s.onnx
```

### Sample Data

Sample input data is available from the same Zenodo repository. Place files in the `data/sample/` directory.

### MongoDB Data

MongoDB data is persisted in a Docker volume (`mongodb_data`). To backup:

```bash
# Export MongoDB data
docker-compose exec mongodb mongodump --out /data/backup

# Import MongoDB data
docker-compose exec mongodb mongorestore /data/backup
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check if NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:11.7.1-base nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory (OOM)

Reduce the number of ensemble members or forecast steps:

```bash
docker-compose run --rm fuxi-s2s pipeline --members 5 --steps 21
```

### MongoDB Connection Failed

Ensure MongoDB is running:

```bash
# Check MongoDB status
docker-compose ps mongodb

# Start MongoDB
docker-compose up -d mongodb

# View MongoDB logs
docker-compose logs mongodb
```

### CDS API Errors

1. Verify your API key is correct
2. Check if ERA5 data is available for the requested date (typically 5-day delay)
3. Ensure the `.cdsapirc` file has correct permissions

### Build Failures

```bash
# Clean up Docker system
docker system prune -af

# Rebuild with no cache
docker-compose build --no-cache
```

---

## Development Mode

For development with Mongo Express web UI:

```bash
# Start all services including Mongo Express
docker-compose --profile dev up -d

# Access Mongo Express at http://localhost:8081
# Username: admin
# Password: fuxi2024
```

---

## Production Deployment

### Running as a Service

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f fuxi-s2s

# Stop services
docker-compose down
```

### Scheduled Forecasts (Cron)

Add to crontab for daily forecasts:

```bash
# Run daily at 6:00 AM
0 6 * * * cd /path/to/FuXi-S2S && docker-compose run --rm fuxi-s2s pipeline >> /var/log/fuxi-s2s.log 2>&1
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│                                                                 │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │   fuxi-s2s          │        │     mongodb         │        │
│  │   Container         │◄──────►│     Container       │        │
│  │                     │        │                     │        │
│  │  - Python 3.10      │        │  - MongoDB 7.0      │        │
│  │  - CUDA 11.7        │        │  - Persistent Vol   │        │
│  │  - PyTorch 2.0      │        │                     │        │
│  │  - ONNX Runtime GPU │        └─────────────────────┘        │
│  │                     │                                        │
│  └─────────────────────┘                                        │
│         │                                                       │
│         │ GPU Access                                            │
│         ▼                                                       │
│  ┌─────────────────────┐                                        │
│  │   NVIDIA GPU        │                                        │
│  │   (Host)            │                                        │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Support

For issues related to:

- **Docker setup**: Check this guide's troubleshooting section
- **FuXi-S2S model**: See the main [README.md](README.md)
- **ERA5 data**: Visit [CDS Copernicus](https://cds.climate.copernicus.eu/)
