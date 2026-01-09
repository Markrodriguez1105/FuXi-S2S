# FuXi-S2S Docker Guide

This guide covers the Dockerized deployment of FuXi-S2S as a microservices architecture.

## 📦 Architecture

The system is organized as two microservices:

```txt
┌──────────────────────────────────────────────────────────────────┐
│                        Docker Network                             │
├───────────────────┬─────────────────────┬────────────────────────┤
│  weather_service  │   fuxis2s_model     │      mongodb           │
│   (Port 5002)     │    (Port 8002)      │    (Port 27017)        │
│                   │                     │                        │
│ • Forecast API    │ • GPU Inference     │ • Forecast Storage     │
│ • Station data    │ • ERA5 Download     │ • Run History          │
│ • External access │ • Data Pipeline     │ • Stations             │
└───────────────────┴─────────────────────┴────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- CDS API credentials (for ERA5 data)

### 2. Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your CDS API credentials
# Get your key from: https://cds.climate.copernicus.eu/

# Start all services
docker-compose up -d
```

### 3. Verify Services

```bash
# Check service health
curl http://localhost:5002/health    # Weather Service
curl http://localhost:8002/health    # Model Service

# View logs
docker-compose logs -f
```

## 🔌 API Endpoints

### Weather Service (Port 5002)

| Endpoint                    | Method | Description          |
| --------------------------- | ------ | -------------------- |
| `/health`                   | GET    | Health check         |
| `/ready`                    | GET    | Readiness check      |
| `/api/v1/forecast/latest`   | GET    | Get latest forecast  |
| `/api/v1/forecast/by-date`  | GET    | Get forecast by date |
| `/api/v1/forecast/runs`     | GET    | List forecast runs   |
| `/api/v1/forecast/stations` | GET    | List stations        |

### Model Service (Port 8002)

| Endpoint                            | Method | Description      |
| ----------------------------------- | ------ | ---------------- |
| `/health`                           | GET    | Health check     |
| `/health/gpu`                       | GET    | GPU status       |
| `/health/model`                     | GET    | Model status     |
| `/api/v1/inference/run`             | POST   | Run inference    |
| `/api/v1/inference/status/{job_id}` | GET    | Job status       |
| `/api/v1/pipeline/run`              | POST   | Full pipeline    |
| `/api/v1/pipeline/download`         | POST   | Download ERA5    |
| `/api/v1/pipeline/store`            | POST   | Store to MongoDB |

## 📁 Project Structure

```txt
FuXi-S2S/
├── weather_service/          # Weather API service
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── Dockerfile           # Container definition
│   ├── requirements.txt     # Dependencies
│   ├── routers/             # API endpoints
│   │   ├── forecast.py      # Forecast endpoints
│   │   └── health.py        # Health checks
│   └── services/            # Business logic
│       └── mongo_client.py  # Database client
│
├── fuxis2s_model/           # Model inference service
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── Dockerfile           # GPU container
│   ├── requirements.txt     # Dependencies
│   ├── docker-entrypoint.sh # Container entrypoint
│   ├── routers/             # API endpoints
│   │   ├── inference.py     # Model inference
│   │   ├── pipeline.py      # Data pipeline
│   │   └── health.py        # Health checks
│   └── core/                # Core logic
│       ├── inference.py     # Model loading & inference
│       ├── download_era5.py # ERA5 data download
│       ├── store_forecasts.py # MongoDB storage
│       ├── data_util.py     # Data utilities
│       └── pipeline.py      # Full workflow
│
├── docker-compose.yml       # Service orchestration
├── docker/                  # Docker support files
│   └── mongo-init/          # MongoDB init scripts
├── model/                   # ONNX model files
├── data/                    # Input/output data
│   ├── realtime/            # ERA5 downloads
│   └── sample/              # Sample data
├── output/                  # Forecast outputs
└── .env.example            # Environment template
```

## 🔧 Configuration

### Environment Variables

| Variable          | Default                  | Description        |
| ----------------- | ------------------------ | ------------------ |
| `CDS_API_URL`     | <https://cds.climate...> | CDS API URL        |
| `CDS_API_KEY`     | -                        | Your CDS API key   |
| `MONGO_DB_URI`    | mongodb://...            | MongoDB connection |
| `MONGO_DB`        | arice                    | Database name      |
| `DEFAULT_MEMBERS` | 11                       | Ensemble members   |
| `DEFAULT_STEPS`   | 42                       | Forecast days      |
| `DEVICE`          | cuda                     | cuda or cpu        |

### Model Settings

The model service can be configured via environment variables:

- `MODEL_PATH`: Path to ONNX model file
- `DATA_DIR`: Input data directory
- `OUTPUT_DIR`: Forecast output directory
- `CROP_LAT/LON/RADIUS`: Regional crop settings

## 🖥️ Usage Examples

### Run Inference via API

```bash
# Start full pipeline
curl -X POST http://localhost:8002/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"members": 11, "total_step": 42}'

# Check job status
curl http://localhost:8002/api/v1/inference/status/JOB_ID
```

### Get Forecast Data

```bash
# Get latest forecast for a station
curl "http://localhost:5002/api/v1/forecast/latest?station=Pacol,%20Naga%20City"

# Get forecast by date
curl "http://localhost:5002/api/v1/forecast/by-date?date=2024-01-15"
```

### Direct Container Access

```bash
# Run inference directly
docker-compose exec fuxis2s_model python -m core.inference \
  --model /app/model/fuxi_s2s.onnx \
  --input /app/data/realtime \
  --save_dir /app/output

# Download ERA5 data
docker-compose exec fuxis2s_model python -m core.download_era5

# Open shell in model container
docker-compose exec fuxis2s_model bash
```

## 📊 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fuxis2s_model
docker-compose logs -f weather_service
```

### Health Checks

```bash
# Check all services
curl http://localhost:5002/health
curl http://localhost:8002/health
curl http://localhost:8002/health/gpu
```

## 🛠️ Development

### Build Images

```bash
# Build all
docker-compose build

# Build specific service
docker-compose build fuxis2s_model
docker-compose build weather_service
```

### Reset MongoDB

```bash
# Stop and remove volumes
docker-compose down -v

# Start fresh
docker-compose up -d
```

## 🔗 External Service Communication

The services are configured to accept connections from external devices:

1. **CORS**: Both services have CORS middleware enabled
2. **Network**: Services are on a bridge network with ports exposed
3. **DNS**: Configure external devices to reach the host machine's IP

Example from another device:

```bash
# Replace 192.168.1.100 with your host's IP
curl http://192.168.1.100:5002/api/v1/forecast/latest
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker info | grep -i nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

### MongoDB Connection Issues

```bash
# Reset MongoDB completely
docker-compose down -v
docker-compose up -d mongodb
docker-compose logs mongodb
```

### Service Won't Start

```bash
# Check logs
docker-compose logs fuxis2s_model
docker-compose logs weather_service

# Rebuild
docker-compose build --no-cache
```
