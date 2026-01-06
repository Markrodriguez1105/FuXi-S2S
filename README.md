## FuXi-S2S

This is the official repository for the FuXi-S2S paper.

A machine learning model that outperforms conventional global subseasonal forecast models

by Lei Chen, Xiaohui Zhong, Jie Wu, Deliang Chen, Shangping Xie, Qingchen Chao, Chensen Lin, Zixin Hu, Bo Lu, Hao Li, Yuan Qi

## Installation

The FuXi-S2S model and sample input data used in this study are now publicly available via Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15718402.svg)](https://zenodo.org/records/15718402)

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── data
│   │    ├── mask.nc
│   │    ├── sample
│   │         ├── geopotential.nc
│   │         ├── temperature.nc
│   │         ├── ......
│   │         ├── total_precipitation.nc
|   |
│   ├── model
│   |    ├── fuxi_s2s.onnx
|   |
│   ├── inference.py
│   ├── data_util.py

```

1. Install xarray

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
```

2. Install pytorch

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Usage

### Quick Start: Automated Full Process

Run the complete workflow (inference → comparison → analysis → visualization) with a single command:

**Windows (PowerShell):**

```powershell
.\run_full_process.ps1
```

**Linux/Mac (Bash):**

```bash
chmod +x run_full_process.sh
./run_full_process.sh
```

**Customization:**

```powershell
# Windows - Custom parameters
.\run_full_process.ps1 -total_step 30 -total_member 11

# Skip certain steps
.\run_full_process.ps1 -skip_inference -skip_comparison

# Linux/Mac - Custom parameters
./run_full_process.sh --total_step 30 --total_member 11

# Skip certain steps
./run_full_process.sh --skip-inference --skip-comparison
```

**Available Parameters:**

- `model`: Path to ONNX model (default: `model/fuxi_s2s.onnx`)
- `input_data`: Path to input data (default: `data/sample`) - Windows only, use `input` for Linux/Mac
- `total_step`: Forecast days (default: `42`)
- `total_member`: Ensemble members (default: `11`)
- `save_dir`: Output directory (default: `output`)
- `skip_inference`: Skip step 1
- `skip_comparison`: Skip step 2
- `skip_analysis`: Skip step 3
- `skip_visualization`: Skip step 4

---

### Manual Process: Step-by-Step

#### Step 1: Run FuXi-S2S Inference

Run the inference script to generate subseasonal forecasts:

```bash
python inference.py \
    --model model/fuxi_s2s.onnx \
    --input data/sample \
    --total_step 42 \
    --total_member 5 \
    --save_dir output
```

**Parameters:**

- `--model`: Path to the ONNX model file
- `--input`: Path to input data directory containing individual variable files
- `--total_step`: Number of forecast days (e.g., 42 for 42-day forecast)
- `--total_member`: Number of ensemble members (e.g., 5 or 11)
- `--save_dir`: Output directory for forecast files

**GPU Acceleration:**
The script automatically uses GPU if available (requires CUDA 12.0 + cuDNN 9.x for onnxruntime-gpu).

- CPU: ~2-5 minutes per day
- GPU (RTX 4050): ~1.7 seconds per day

**Output Structure:**

```
output/
├── YYYY/
│   ├── YYYYMMDD/
│   │   ├── member/
│   │   │   ├── 01/
│   │   │   │   ├── 01.nc
│   │   │   │   ├── 02.nc
│   │   │   │   └── ...
│   │   │   ├── 02/
│   │   │   └── ...
```

### Step 2: Compare with PAGASA Observations (Optional)

Compare forecasts against PAGASA weather station data:

```bash
python compare_pagasa.py --fuxi_output output --member 0
```

Run comparison for all ensemble members:

```bash
for ($i=0; $i -lt 5; $i++) {
    python compare_pagasa.py --fuxi_output output --member $i
}
```

**Requirements:**

- PAGASA data file: `data/PAGASA_CBSUA_Pili_2020_2023.xlsx`
- Columns: YEAR, MONTH, DAY, WIND DIRECTION, WINDSPEED, TMAX, TMIN, RAINFALL

**Output:**
Comparison CSV files saved to `compare/result/CBSUA_Pili/YYYYMM.csv` with:

- Observed vs predicted values for rainfall, temperature, wind speed, wind direction
- Error metrics (RMSE, MAE, Bias)

### Step 3: Analyze Results (Optional)

Generate statistical analysis summary:

```bash
python analyze_results.py
```

**Output:** `ANALYSIS_SUMMARY.txt` containing:

- RMSE, MAE, and Bias for all variables
- Correlation coefficients
- Error analysis by lead time
- Interpretation guide for metrics

### Step 4: Visualize Results (Optional)

Create comprehensive visualization plots:

```bash
python visualize_results.py
```

**Output:** 6 PNG plots in `compare/plots/`:

1. Rainfall time series (observed vs predicted)
2. Temperature time series (TMAX vs t2m)
3. Rainfall scatter plot with perfect prediction line
4. Temperature scatter plot
5. Error by lead time (1-42 days)
6. Error distribution histograms

## Wind Speed and Direction

The model predicts 10m wind components (10u, 10v) which are automatically converted to:

- **Wind Speed** (m/s): √(10u² + 10v²)
- **Wind Direction** (0-360°): Meteorological convention (0° = from North, clockwise)

These are included in the comparison outputs and visualizations.

## Input preparation

The `input.nc` file contains preprocessed data from the origin ERA5 files. The file has a shape of (2, 76, 121, 240), where the first dimension represents two time steps. The second dimension represents all variable and level combinations, named in the following exact order:

```python
['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300',
'z250', 'z200', 'z150', 'z100', 'z50', 't1000', 't925', 't850',
't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150',
't100', 't50', 'u1000', 'u925', 'u850', 'u700', 'u600', 'u500',
'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50', 'v1000',
'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250',
'v200', 'v150', 'v100', 'v50', 'q1000', 'q925', 'q850', 'q700',
'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100',
'q50', 't2m', 'd2m', 'sst', 'ttr', '10u', '10v', '100u', '100v',
'msl', 'tcwv', 'tp']
```

The last 11 variables: ('t2m', 'd2m', 'sst', 'ttr', '10u', '10v', '100u', '100v',
'msl', 'tcwv', 'tp') are surface variables, while the remaining variables represent atmosphere variables with numbers representing pressure levels.
