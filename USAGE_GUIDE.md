# FuXi-S2S Usage Guide

## Quick Reference

### Project Structure

```
FuXi-S2S/
├── data/
│   ├── mask.nc               # Land/sea mask
│   ├── sample/               # Sample ERA5 data files (individual variables)
│   └── PAGASA_CBSUA_Pili_2020_2023.xlsx  # PAGASA observations
├── model/
│   └── fuxi_s2s.onnx        # Model file (2.1 GB, download from Zenodo)
├── output/                   # Forecast outputs
│   └── YYYY/                 # Year folder (e.g., 2020/)
│       └── YYYYMMDD/         # Date folder (e.g., 20200602/)
│           └── member/       # Ensemble members
│               └── XX/       # Member number (00-10)
│                   ├── 01.nc # Lead time files
│                   ├── 02.nc
│                   └── ...
├── compare/                  # Comparison results
│   ├── result/
│   │   └── CBSUA_Pili/      # Station-specific results
│   │       ├── 202006.csv   # Monthly comparison files
│   │       ├── 202007.csv
│   │       └── ...
│   └── plots/                # Visualization outputs
│       ├── rainfall_timeseries.png
│       ├── temperature_timeseries.png
│       └── ...
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── display.py           # Console formatting
│   ├── model.py             # Model loading
│   ├── data.py              # Data processing
│   ├── inference.py         # Inference logic
│   └── compare.py           # PAGASA comparison
│   └── mongo_store.py        # MongoDB persistence helpers
├── venv_fuxi/               # Virtual environment
├── inference.py             # Main inference script
├── compare_pagasa.py        # Comparison wrapper
├── analyze_results.py       # Statistical analysis
├── visualize_results.py     # Visualization script
├── store_forecasts_to_mongo.py  # Store station forecasts to MongoDB
├── MONGODB_SETUP_AND_STORAGE_GUIDE.md  # MongoDB setup + schema
├── .env                     # Local env vars (e.g., MONGO_DB_URI)
├── run_full_process.ps1     # Automated workflow (Windows)
├── run_full_process.sh      # Automated workflow (Linux/Mac)
└── data_util.py             # Legacy data utilities
```

---

## Quick Start (Recommended)

### Automated Full Workflow

Run the complete process with a single command:

**Windows (PowerShell):**

```powershell
# Activate environment first
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# Run full process (inference → comparison → analysis → visualization)
.\run_full_process.ps1

# Custom configuration
.\run_full_process.ps1 -total_step 30 -total_member 5

# Skip certain steps
.\run_full_process.ps1 -skip_inference  # Use existing outputs
```

**Linux/Mac (Bash):**

```bash
# Make executable
chmod +x run_full_process.sh

# Run full process
./run_full_process.sh

# Custom configuration
./run_full_process.sh --total_step 30 --total_member 5

# Skip certain steps
./run_full_process.sh --skip-inference
```

**Available Parameters:**

- `model`: Path to ONNX model (default: `model/fuxi_s2s.onnx`)
- `input_data` (Windows) / `input` (Linux): Input directory (default: `data/sample`)
- `total_step`: Forecast days (default: `42`)
- `total_member`: Ensemble members (default: `11`)
- `save_dir`: Output directory (default: `output`)
- `skip_inference`, `skip_comparison`, `skip_analysis`, `skip_visualization`

**What the script does:**

1. ✅ Runs FuXi-S2S inference with GPU acceleration
2. ✅ Compares forecasts with PAGASA observations (all members)
3. ✅ Generates statistical analysis (RMSE, MAE, Bias)
4. ✅ Creates 6 comprehensive visualization plots
5. ✅ Opens results folder automatically

---

## Manual Process

### Setup

### 1. Virtual Environment

```powershell
# Activate virtual environment
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# Verify activation (should see (venv_fuxi) in prompt)
```

### 2. Download Model

Download `fuxi_s2s.onnx` from [Zenodo](https://zenodo.org/records/15718402) and place in `model/` folder.

```powershell
# Model should be at:
# C:\Machine Learning\FuXi-S2S\model\fuxi_s2s.onnx
```

### 3. Input Data

The `data/sample/` directory contains individual NetCDF files for each variable:

- `geopotential.nc`, `temperature.nc`, `u_component_of_wind.nc`, etc.
- These files are automatically loaded by the inference script
- No preprocessing needed - use `data/sample` directly as input

```powershell
# Verify sample data exists
ls data/sample/*.nc
```

---

## Running Inference

### Lightweight Testing Mode (Recommended for Testing)

Fast execution: 7 forecast days × 2 ensemble members = 14 total steps

```powershell
python inference.py `
    --model model/fuxi_s2s.onnx `
    --input data/sample `
    --save_dir output `
    --lite
```

### Custom Configuration

```powershell
python inference.py `
    --model model/fuxi_s2s.onnx `
    --input data/sample `
    --save_dir output `
    --total_step 5 `
    --total_member 1
```

### Full Production Run

42 forecast days × 11 ensemble members = 462 total steps (~1-2 hours with GPU, ~8+ hours with CPU)

```powershell
python inference.py `
    --model model/fuxi_s2s.onnx `
    --input data/sample `
    --save_dir output `
    --total_step 42 `
    --total_member 11
```

### GPU Acceleration

The script automatically uses GPU if available (requires CUDA 12.0 + cuDNN 9.x):

- **GPU (RTX 4050)**: ~1.7 seconds per day (~50-100x faster)
- **CPU**: ~2-5 minutes per day

### Using CPU Instead of GPU

```powershell
python inference.py `
    --model model/fuxi_s2s.onnx `
    --input data/sample `
    --save_dir output `
    --device cpu `
    --lite
```

---

## Comparing with PAGASA Data

### Basic Comparison (Auto-detect latest forecast)

```powershell
python compare_pagasa.py `
    --fuxi_output output `
    --member 0
```

### Specify Initialization Date

```powershell
# Format: YYYYMMDD
python compare_pagasa.py `
    --fuxi_output output `
    --init_date 20200103 `
    --member 0

# Alternative formats also work:
# --init_date 2020/01/03
# --init_date 2020-01-03
```

### Compare Different Ensemble Members

```powershell
# Member 0 (first ensemble member)
python compare_pagasa.py --fuxi_output output --init_date 20200103 --member 0

# Member 5
python compare_pagasa.py --fuxi_output output --init_date 20200103 --member 5
```

### Custom Station Coordinates

```powershell
python compare_pagasa.py `
    --fuxi_output output `
    --init_date 20200103 `
    --lat 13.62 `
    --lon 123.19 `
    --member 0
```

### Custom Output File

```powershell
python compare_pagasa.py `
    --fuxi_output output `
    --init_date 20200103 `
    --member 0 `
    --output results_member00.csv
```

---

## Store Forecasts to MongoDB

Persist station-level forecasts to MongoDB for downstream use (dashboards, APIs).

```powershell
python store_forecasts_to_mongo.py --fuxi_output output --init_date 20200602 --station "CBSUA Pili" --members 11
```

| Option          | Description                                    |
| --------------- | ---------------------------------------------- |
| `--fuxi_output` | Path to output directory (default: `output`)   |
| `--init_date`   | Initialization date `YYYYMMDD`                 |
| `--station`     | Station name from `utils/compare.py::STATIONS` |
| `--members`     | Number of ensemble members                     |
| `--version`     | Optional version suffix for the run            |

See [MONGODB_SETUP_AND_STORAGE_GUIDE.md](MONGODB_SETUP_AND_STORAGE_GUIDE.md) for full details.

## Available Stations

Predefined in `compare_pagasa.py`:

- **CBSUA Pili**: 13.58°N, 123.28°E (default)
- **Naga City**: 13.62°N, 123.19°E

Use with `--station` flag:

```powershell
python compare_pagasa.py --station "Naga City" --member 0
```

---

## Output Files

### Inference Output Structure

```
output/
└── 2020/
    └── 20200103/
        └── member/
            ├── 00/
            │   ├── 01.nc  # Day 1 forecast
            │   ├── 02.nc  # Day 2 forecast
            │   └── ...
            ├── 01/
            └── ...
```

### Comparison Results

Output directory: `compare/result/CBSUA_Pili/YYYYMM.csv` (monthly files)

Columns include:

**PAGASA Observations:**

- `YEAR`, `MONTH`, `DAY`, `Date`: Observation date
- `RAINFALL`: Observed rainfall (mm)
- `TMAX`, `TMIN`: Observed temperatures (°C)
- `WIND DIRECTION`: Observed wind direction (0-360°)
- `WINDSPEED`: Observed wind speed (m/s)

**FuXi Forecasts:**

- `init_time`: Forecast initialization date
- `valid_time`: Forecast valid time
- `lead_time_days`: Forecast lead time (1-42 days)
- `tp`: Forecasted precipitation (mm)
- `t2m`: 2m temperature (K)
- `t2m_celsius`: 2m temperature (°C)
- `10u`, `10v`: 10m wind components (m/s)
- `wind_speed`: Calculated wind speed from √(u²+v²) (m/s)
- `wind_direction`: Calculated direction from atan2(u,v) (0-360°)
- `msl`: Mean sea level pressure (Pa)

**Error Metrics:**

- `precip_error`, `precip_abs_error`: Rainfall errors
- `temp_error`: Temperature error (TMAX - t2m)
- `tmin_error`: TMIN vs t2m error
- `wind_dir_error`: Wind direction circular difference (°)

---

## Statistical Analysis

### Generate Analysis Summary

```powershell
python analyze_results.py
```

**Output:** `ANALYSIS_SUMMARY.txt`

**Includes:**

- Overall statistics (RMSE, MAE, Bias) for:
  - Rainfall (mm)
  - Temperature (°C)
  - Wind speed (m/s)
  - Wind direction (circular MAE in degrees)
- Correlation coefficients
- Error analysis by lead time
- Interpretation guide for metrics

**Sample Output:**

```
FORECAST ACCURACY SUMMARY
==================================================
Analysis Date: 2026-01-07
Forecast Points Analyzed: 42
Date Range: 2020-06-03 to 2020-07-14

PRECIPITATION
  RMSE: 17.87 mm
  MAE:  7.92 mm
  Bias: +7.62 mm (model underestimates)

TEMPERATURE (TMAX vs t2m)
  RMSE: 4.83°C
  MAE:  4.44°C
  Bias: +4.41°C (model underestimates)

WIND DIRECTION
  MAE:  88.0° (circular difference)
```

---

## Visualization

### Create Plots

```powershell
python visualize_results.py
```

**Output:** 6 PNG files in `compare/plots/`

1. **rainfall_timeseries.png**: Observed vs predicted rainfall over time
2. **temperature_timeseries.png**: TMAX vs t2m over time
3. **rainfall_scatter.png**: Scatter plot with perfect prediction line
4. **temperature_scatter.png**: Temperature scatter plot
5. **error_by_leadtime.png**: How errors change with forecast lead time (1-42 days)
6. **error_distribution.png**: Histograms of rainfall and temperature errors

**View Results:**

```powershell
# Open plots folder
explorer compare\plots

# Or on Linux/Mac
xdg-open compare/plots
```

---

## Wind Speed and Direction

The model predicts 10m wind components (10u, 10v) which are automatically converted to:

**Wind Speed (m/s):**

```python
wind_speed = sqrt(10u² + 10v²)
```

**Wind Direction (0-360°, meteorological convention):**

```python
wind_direction = atan2(10u, 10v) converted to degrees
# 0° = from North, clockwise (90° = from East, 180° = from South, 270° = from West)
```

Wind predictions are included in:

- Comparison CSV files (`wind_speed`, `wind_direction` columns)
- Analysis summary (Wind Direction MAE)
- Can be visualized by modifying `visualize_results.py`

---

## Common Workflows

### 1. Automated Full Process (Recommended)

```powershell
# Activate environment
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# Run everything (inference → comparison → analysis → visualization)
.\run_full_process.ps1

# Custom: 30 days, 5 members
.\run_full_process.ps1 -total_step 30 -total_member 5

# Skip inference if already done
.\run_full_process.ps1 -skip_inference
```

### 2. Quick Manual Test

```powershell
# Activate environment
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# Run lightweight inference
python inference.py --model model/fuxi_s2s.onnx --input data/sample --save_dir output --lite

# Compare with PAGASA
python compare_pagasa.py --fuxi_output output --member 0

# Analyze
python analyze_results.py

# Visualize
python visualize_results.py
```

### 3. Full Production Run (Manual)

```powershell
# Activate environment
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# Run full inference (~1-2 hours with GPU)
python inference.py --model model/fuxi_s2s.onnx --input data/sample --save_dir output --total_step 42 --total_member 11

# Compare all members
for ($i=0; $i -lt 11; $i++) {
    python compare_pagasa.py --fuxi_output output --member $i
}

# Analyze and visualize
python analyze_results.py
python visualize_results.py
```

### 4. Compare Different Members

```powershell
# Compare specific ensemble members
python compare_pagasa.py --fuxi_output output --member 0  # First member
python compare_pagasa.py --fuxi_output output --member 5  # Sixth member
python compare_pagasa.py --fuxi_output output --member 10 # Last member
```

---

## Command-Line Arguments Reference

### inference.py

| Argument         | Type  | Default      | Description                                   |
| ---------------- | ----- | ------------ | --------------------------------------------- |
| `--model`        | str   | **required** | Path to FuXi-S2S ONNX model                   |
| `--input`        | str   | **required** | Path to input directory (e.g., `data/sample`) |
| `--device`       | str   | `cuda`       | Device: `cuda` or `cpu`                       |
| `--save_dir`     | str   | `output`     | Output directory                              |
| `--total_step`   | int   | `42`         | Number of forecast days                       |
| `--total_member` | int   | `1`          | Number of ensemble members                    |
| `--lite`         | flag  | -            | Lightweight mode (7 steps, 2 members)         |
| `--crop_lat`     | float | -            | Center latitude for cropping                  |
| `--crop_lon`     | float | -            | Center longitude for cropping                 |
| `--crop_radius`  | float | -            | Radius in degrees for regional cropping       |

### compare_pagasa.py

| Argument        | Type  | Default                                 | Description                     |
| --------------- | ----- | --------------------------------------- | ------------------------------- |
| `--pagasa`      | str   | `data/PAGASA_CBSUA_Pili_2020_2023.xlsx` | PAGASA Excel file path          |
| `--fuxi_output` | str   | `output`                                | FuXi output directory           |
| `--member`      | int   | `0`                                     | Ensemble member (0-10)          |
| `--init_date`   | str   | `None` (auto-detect)                    | Init date (YYYYMMDD/YYYY/MM/DD) |
| `--station`     | str   | `CBSUA Pili`                            | Station name                    |
| `--lat`         | float | `None`                                  | Override latitude               |
| `--lon`         | float | `None`                                  | Override longitude              |

**Note:** Output is automatically saved to `compare/result/[station]/YYYYMM.csv` (monthly files)

### run_full_process.ps1 / run_full_process.sh

| Parameter (PS)        | Parameter (Bash)       | Default               | Description             |
| --------------------- | ---------------------- | --------------------- | ----------------------- |
| `-model`              | `--model`              | `model/fuxi_s2s.onnx` | Path to ONNX model      |
| `-input_data`         | `--input`              | `data/sample`         | Input data directory    |
| `-total_step`         | `--total_step`         | `42`                  | Forecast days           |
| `-total_member`       | `--total_member`       | `11`                  | Ensemble members        |
| `-save_dir`           | `--save_dir`           | `output`              | Output directory        |
| `-skip_inference`     | `--skip-inference`     | -                     | Skip inference step     |
| `-skip_comparison`    | `--skip-comparison`    | -                     | Skip comparison step    |
| `-skip_analysis`      | `--skip-analysis`      | -                     | Skip analysis step      |
| `-skip_visualization` | `--skip-visualization` | -                     | Skip visualization step |

---

## Troubleshooting

### Model file not found

```
Error: Load model from model/fuxi_s2s.onnx failed
```

**Solution**: Download model from Zenodo and place in `model/` folder.

### Input data issues

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/sample/geopotential.nc'
```

**Solution**: Ensure all sample data files are in `data/sample/` directory. Download from Zenodo if missing.

### CUDA/GPU errors

```
Error: CUDAExecutionProvider not available
```

**Solution**: Use CPU instead:

```powershell
python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --save_dir output --device cpu --lite
```

### Member directory not found

```
Member directory not found: output/2020/20200103/member/00
```

**Solution**: Run inference first, then compare:

```powershell
python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --save_dir output --lite
python compare_pagasa.py --fuxi_output output --member 0
```

---

## Performance Tips

1. **Use automated script** (`run_full_process.ps1`) for complete workflow
2. **Start with `--lite` flag** for testing individual steps
3. **Use GPU** (CUDA 12.0 + cuDNN 9.x) for faster inference:
   - GPU (RTX 4050): ~1.7 seconds per forecast day (~50-100x faster)
   - CPU: ~2-5 minutes per forecast day
4. **Expected runtimes with GPU:**
   - Lite mode (7 days × 2 members): ~30 seconds
   - Custom (30 days × 5 members): ~4-5 minutes
   - Full run (42 days × 11 members): ~12-15 minutes
5. **Expected runtimes with CPU:**
   - Lite mode: ~20-30 minutes
   - Full run: ~8-20 hours
6. **Monitor progress**: Each step prints timing and progress information
7. **Regional cropping**: Use `--crop_lat`, `--crop_lon`, `--crop_radius` to reduce computational area

---

## Data Sources

- **FuXi-S2S Model**: [Zenodo](https://zenodo.org/records/15718402)
- **ERA5 Sample Data**: Included in repository
- **PAGASA Data**: Local observations from CBSUA Pili station

---

## Citation

If using FuXi-S2S, please cite:

```bibtex
@article{chen2024machine,
  title={A machine learning model that outperforms conventional global subseasonal forecast models},
  author={Chen, Lei and Zhong, Xiaohui and Li, Hao and Wu, Jie and Lu, Bo and Chen, Deliang and Xie, Shang-Ping and Wu, Libo and Chao, Qingchen and Lin, Chensen and others},
  journal={Nature Communications},
  volume={15},
  number={6425},
  pages={1-14},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={https://doi.org/10.1038/s41467-024-50714-1}
}
```

---

## Quick Command Cheat Sheet

```powershell
# Activate environment
& "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\Activate.ps1"

# RECOMMENDED: Automated full process
.\run_full_process.ps1

# Test run (manual)
python inference.py --model model/fuxi_s2s.onnx --input data/sample --save_dir output --lite
python compare_pagasa.py --fuxi_output output --member 0
python analyze_results.py
python visualize_results.py

# Full run (manual)
python inference.py --model model/fuxi_s2s.onnx --input data/sample --save_dir output --total_step 42 --total_member 11
for ($i=0; $i -lt 11; $i++) { python compare_pagasa.py --fuxi_output output --member $i }
python analyze_results.py
python visualize_results.py

# View results
explorer compare\plots
cat ANALYSIS_SUMMARY.txt
```
