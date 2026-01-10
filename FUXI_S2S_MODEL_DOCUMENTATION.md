# FuXi-S2S: Subseasonal-to-Seasonal Weather Forecast Model

## Technical Documentation for Research and Development

---

## Table of Contents

1. [Introduction](#introduction)
2. [Model Architecture Overview](#model-architecture-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Algorithm Description](#algorithm-description)
5. [Input Data Structure](#input-data-structure)
6. [Ensemble Forecasting](#ensemble-forecasting)
7. [Output Data Structure](#output-data-structure)
8. [Wind Speed and Direction Calculations](#wind-speed-and-direction-calculations)
9. [Pipeline Workflow](#pipeline-workflow)
10. [Configuration Parameters](#configuration-parameters)
11. [References](#references)

---

## Introduction

**FuXi-S2S** (pronounced "Fu-Xi Subseasonal-to-Seasonal") is a state-of-the-art machine learning model designed for **subseasonal-to-seasonal (S2S) weather forecasting**. The model outperforms conventional global subseasonal forecast models by leveraging deep learning architectures trained on ERA5 reanalysis data.

### Key Features

- **Forecast Horizon**: Up to 42 days (6 weeks)
- **Ensemble Members**: Configurable (default: 11 members)
- **Spatial Resolution**: 1.5° global grid (121 × 240 points)
- **Temporal Resolution**: Daily forecasts
- **Model Format**: ONNX (Open Neural Network Exchange)

### Authors

Lei Chen, Xiaohui Zhong, Jie Wu, Deliang Chen, Shangping Xie, Qingchen Chao, Chensen Lin, Zixin Hu, Bo Lu, Hao Li, Yuan Qi

---

## Model Architecture Overview

FuXi-S2S employs a **deep neural network** architecture optimized for spatiotemporal weather prediction. The model is deployed as an ONNX model, enabling efficient inference on both CPU and GPU hardware.

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    FuXi-S2S Neural Network                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   INPUT LAYER                                                   │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Shape: (batch, 2, 76, 121, 240)                         │  │
│   │ - 2 time steps (t-1, t)                                 │  │
│   │ - 76 channels (variables × levels)                      │  │
│   │ - 121 latitude points (90°N to 90°S)                    │  │
│   │ - 240 longitude points (0° to 360°)                     │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│   ENCODER (Spatiotemporal Feature Extraction)                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ - Multi-scale convolutions                              │  │
│   │ - Attention mechanisms                                  │  │
│   │ - Residual connections                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│   TEMPORAL PROCESSOR                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ - Day-of-year embedding (doy)                           │  │
│   │ - Step embedding (forecast lead time)                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│   DECODER (Prediction Generation)                              │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Shape: (batch, 2, 76, 121, 240)                         │  │
│   │ Output: Next timestep atmospheric state                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### State Transition Formula

The FuXi-S2S model predicts the atmospheric state at time $t+1$ based on states at times $t-1$ and $t$:

$$
\mathbf{X}_{t+1} = \mathcal{F}(\mathbf{X}_{t-1}, \mathbf{X}_t, \phi, \text{doy}_t)
$$

Where:

- $\mathbf{X}_t \in \mathbb{R}^{C \times H \times W}$ is the atmospheric state at time $t$
- $C = 76$ is the number of channels (variables × pressure levels)
- $H = 121$ is the latitude dimension
- $W = 240$ is the longitude dimension
- $\mathcal{F}$ is the neural network function
- $\phi$ is the step embedding (forecast lead time)
- $\text{doy}_t$ is the day-of-year embedding

### Day-of-Year Encoding

The model uses normalized day-of-year to capture seasonal patterns:

$$
\text{doy} = \frac{\min(d, 365)}{365}
$$

Where $d$ is the day of year (1-366).

### Autoregressive Prediction

For multi-day forecasts, the model operates autoregressively:

$$
\hat{\mathbf{X}}_{t+k} = \mathcal{F}(\hat{\mathbf{X}}_{t+k-2}, \hat{\mathbf{X}}_{t+k-1}, k, \text{doy}_{t+k-1})
$$

Where:

- $k$ is the forecast lead time (1 to 42 days)
- $\hat{\mathbf{X}}$ denotes predicted states

### Normalization

Input data is normalized using ERA5 climatological statistics:

$$
\mathbf{X}_{norm} = \frac{\mathbf{X} - \mu}{\sigma}
$$

Where:

- $\mu$ is the channel-wise mean
- $\sigma$ is the channel-wise standard deviation

---

## Algorithm Description

### Main Inference Algorithm

```
Algorithm: FuXi-S2S Forecast Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:
    model         : ONNX neural network model
    input_data    : DataArray with shape (2, 76, 121, 240)
    total_step    : Number of forecast days (default: 42)
    total_member  : Number of ensemble members (default: 11)
    save_dir      : Output directory

OUTPUT:
    Forecast files for each member and lead time

PROCEDURE:

1. INITIALIZE
   load_model(model_path, device)
   input_data ← load_input_data(input_path)
   init_time ← extract_init_time(input_data)

2. VALIDATE INPUT
   ASSERT lat[0] = 90° AND lat[-1] = -90°
   ASSERT time[1] - time[0] = 1 day

3. FOR member = 0 TO total_member - 1 DO

   3.1 COPY input state
       current_input ← deep_copy(input_data)

   3.2 FOR step = 0 TO total_step - 1 DO

       lead_time ← step + 1

       3.2.1 PREPARE INPUTS
             inputs ← {'input': current_input}

             IF model uses step embedding:
                inputs['step'] ← step

             IF model uses day-of-year embedding:
                valid_time ← init_time + step days
                doy ← min(365, day_of_year(valid_time)) / 365
                inputs['doy'] ← doy

       3.2.2 RUN INFERENCE
             output ← model.run(inputs)

       3.2.3 UPDATE STATE
             current_input ← output
             forecast ← output[:, -1:]  # Extract last time step

       3.2.4 SAVE FORECAST
             save_forecast(forecast, member, lead_time, save_dir)

   END FOR (step)

END FOR (member)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Data Preparation Algorithm

```
Algorithm: Make Input from ERA5 Variables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:
    data_dir : Directory containing individual variable .nc files

OUTPUT:
    DataArray with shape (2, 76, 121, 240)

PROCEDURE:

1. DEFINE variable list (16 variables total)
   pressure_vars ← [z, t, u, v, q]
   surface_vars ← [t2m, d2m, sst, ttr, 10u, 10v, 100u, 100v, msl, tcwv, tp]

2. DEFINE pressure levels (13 levels)
   levels ← [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

3. FOR each variable IN variable_list DO

   3.1 LOAD file
       data ← load_netcdf(data_dir + "/" + variable + ".nc")

   3.2 NORMALIZE dimensions
       rename valid_time → time
       rename pressure_level → level
       rename latitude → lat
       rename longitude → lon

   3.3 REGRID if necessary
       IF grid_size = (721, 1440):  # 0.25° resolution
          data ← downsample(data, factor=6)  # → 1.5° resolution

   3.4 APPLY transformations
       IF variable = 'tp':
          data ← clip(data × 1000, 0, 1000)  # Convert to mm
       IF variable = 'ttr':
          data ← data / 3600  # Convert to hourly rate

   3.5 REORDER levels (descending: 1000 → 50)

   3.6 CREATE channel labels
       IF is_pressure_variable:
          channels ← [var + level for level in levels]
       ELSE:
          channels ← [var]

END FOR

4. CONCATENATE all channels
   combined ← concat(all_variables, dim='channel')
   combined ← transpose(combined, ['time', 'channel', 'lat', 'lon'])

5. RETURN combined

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Input Data Structure

### Input Tensor Shape

```
Shape: (time=2, channel=76, lat=121, lon=240)
```

### Channel Organization

The 76 channels are organized as follows:

| Index Range | Variable                  | Short Name | Levels  | Count |
| ----------- | ------------------------- | ---------- | ------- | ----- |
| 0-12        | Geopotential              | z          | 1000→50 | 13    |
| 13-25       | Temperature               | t          | 1000→50 | 13    |
| 26-38       | U-Wind                    | u          | 1000→50 | 13    |
| 39-51       | V-Wind                    | v          | 1000→50 | 13    |
| 52-64       | Specific Humidity         | q          | 1000→50 | 13    |
| 65          | 2m Temperature            | t2m        | Surface | 1     |
| 66          | 2m Dewpoint               | d2m        | Surface | 1     |
| 67          | Sea Surface Temp          | sst        | Surface | 1     |
| 68          | Top Net Thermal Rad       | ttr        | Surface | 1     |
| 69          | 10m U-Wind                | 10u        | Surface | 1     |
| 70          | 10m V-Wind                | 10v        | Surface | 1     |
| 71          | 100m U-Wind               | 100u       | Surface | 1     |
| 72          | 100m V-Wind               | 100v       | Surface | 1     |
| 73          | Mean Sea Level Pressure   | msl        | Surface | 1     |
| 74          | Total Column Water Vapour | tcwv       | Surface | 1     |
| 75          | Total Precipitation       | tp         | Surface | 1     |

### Pressure Levels (13 levels, in hPa)

```
[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
```

### Spatial Grid

- **Latitude**: 90°N to 90°S (1.5° resolution, 121 points)
- **Longitude**: 0° to 358.5°E (1.5° resolution, 240 points)

---

## Ensemble Forecasting

### Ensemble Generation

FuXi-S2S generates ensemble forecasts by running multiple independent forward passes:

$$
\{\hat{\mathbf{X}}^{(m)}_{t+k}\}_{m=1}^{M}
$$

Where $M$ is the total number of ensemble members (default: 11).

### Ensemble Statistics

#### Ensemble Mean

$$
\bar{\mathbf{X}}_{t+k} = \frac{1}{M} \sum_{m=1}^{M} \hat{\mathbf{X}}^{(m)}_{t+k}
$$

#### Ensemble Spread (Standard Deviation)

$$
\sigma_{t+k} = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} (\hat{\mathbf{X}}^{(m)}_{t+k} - \bar{\mathbf{X}}_{t+k})^2}
$$

#### Probability Forecasts

For a threshold $\tau$:

$$
P(\mathbf{X}_{t+k} > \tau) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}(\hat{\mathbf{X}}^{(m)}_{t+k} > \tau)
$$

---

## Output Data Structure

### File Organization

```
output/
├── YYYY/                          # Year
│   ├── YYYYMMDD/                  # Initialization date
│   │   ├── member/
│   │   │   ├── member00_lead01.nc
│   │   │   ├── member00_lead02.nc
│   │   │   ├── ...
│   │   │   ├── member00_lead42.nc
│   │   │   ├── member01_lead01.nc
│   │   │   └── ...
```

### NetCDF File Structure

Each output file contains:

```
Dimensions:
    channel: 76
    lat: variable (depends on crop settings)
    lon: variable (depends on crop settings)

Coordinates:
    channel: ['z1000', 'z925', ..., 'tp']
    lat: [lat_values]
    lon: [lon_values]

Attributes:
    init_time: "YYYY-MM-DD HH:MM:SS"
    valid_time: "YYYY-MM-DD HH:MM:SS"
    lead_time: integer (1-42)
    member: integer (0-10)
```

---

## Wind Speed and Direction Calculations

### Wind Speed

From 10m wind components (10u, 10v):

$$
\text{Wind Speed} = \sqrt{u_{10}^2 + v_{10}^2} \quad \text{[m/s]}
$$

### Wind Direction

Using meteorological convention (direction wind is coming FROM):

$$
\theta = \left(\arctan2(-u_{10}, -v_{10}) \times \frac{180}{\pi} + 360\right) \mod 360 \quad \text{[degrees]}
$$

Where:

- 0° = Wind from North
- 90° = Wind from East
- 180° = Wind from South
- 270° = Wind from West

### Wind Direction Formula (Alternative)

$$
\theta = \arctan2(u_{10}, v_{10}) \times \frac{180}{\pi} + 180
$$

---

## Pipeline Workflow

### Complete Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    FuXi-S2S COMPLETE PIPELINE                      │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 1: ERA5 DATA DOWNLOAD                                        │
├────────────────────────────────────────────────────────────────────┤
│  • Connect to Copernicus Climate Data Store (CDS)                  │
│  • Download 2 consecutive days of data (t-1, t)                    │
│  • Variables: 5 pressure-level + 11 surface                        │
│  • Levels: 13 pressure levels (1000-50 hPa)                        │
│  • Grid: 0.25° → regrid to 1.5°                                    │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA PREPROCESSING                                        │
├────────────────────────────────────────────────────────────────────┤
│  • Combine individual variable files                               │
│  • Normalize dimensions (time, channel, lat, lon)                  │
│  • Apply unit conversions (tp → mm, ttr → hourly)                  │
│  • Regrid from 0.25° to 1.5° resolution                           │
│  • Create 76-channel input tensor                                  │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 3: MODEL INFERENCE                                           │
├────────────────────────────────────────────────────────────────────┤
│  • Load ONNX model (GPU/CPU)                                       │
│  • For each ensemble member (0 to M-1):                           │
│      • For each forecast day (1 to 42):                           │
│          • Compute day-of-year embedding                          │
│          • Run neural network forward pass                        │
│          • Update autoregressive input                            │
│          • Save daily forecast to NetCDF                          │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 4: POST-PROCESSING & STORAGE                                 │
├────────────────────────────────────────────────────────────────────┤
│  • Extract station-level forecasts                                 │
│  • Convert units (K → °C, etc.)                                   │
│  • Calculate derived variables (wind speed/direction)             │
│  • Store to MongoDB for application access                        │
│  • Generate comparison with observations                          │
└────────────────────────────────────────────────────────────────────┘
```

### Pipeline Timing (GPU RTX 4050)

| Step               | Time   | Description                    |
| ------------------ | ------ | ------------------------------ |
| Model Load         | ~5s    | Load ONNX model to GPU memory  |
| Per Day Inference  | ~1.7s  | Single forward pass            |
| 42-Day Forecast    | ~71s   | Complete forecast for 1 member |
| 11-Member Ensemble | ~13min | Full ensemble generation       |

---

## Configuration Parameters

### Model Configuration

| Parameter         | Default               | Description                        |
| ----------------- | --------------------- | ---------------------------------- |
| `model_path`      | `model/fuxi_s2s.onnx` | Path to ONNX model file            |
| `device`          | `cuda`                | Inference device (`cuda` or `cpu`) |
| `default_members` | 11                    | Number of ensemble members         |
| `default_steps`   | 42                    | Forecast lead time (days)          |

### Regional Crop Settings

| Parameter     | Default | Description                          |
| ------------- | ------- | ------------------------------------ |
| `crop_lat`    | 13.58   | Center latitude for regional output  |
| `crop_lon`    | 123.28  | Center longitude for regional output |
| `crop_radius` | 10.0    | Radius in degrees for regional crop  |

### Data Directories

| Parameter    | Default       | Description               |
| ------------ | ------------- | ------------------------- |
| `data_dir`   | `/app/data`   | Input data directory      |
| `output_dir` | `/app/output` | Forecast output directory |

---

## Verification Metrics

### Root Mean Square Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

### Bias

$$
\text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

### Correlation Coefficient

$$
r = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2 \sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}
$$

---

## References

1. **FuXi-S2S Paper**: Chen, L., Zhong, X., Wu, J., et al. "A machine learning model that outperforms conventional global subseasonal forecast models."

2. **ERA5 Reanalysis**: Hersbach, H., et al. (2020). "The ERA5 global reanalysis." Quarterly Journal of the Royal Meteorological Society.

3. **ONNX Runtime**: Microsoft. "ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator."

4. **Model Repository**: [Zenodo DOI: 10.5281/zenodo.15718402](https://zenodo.org/records/15718402)

---

## Appendix: Channel Names Reference

```python
CHANNELS = [
    # Geopotential (13 levels)
    'z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400',
    'z300', 'z250', 'z200', 'z150', 'z100', 'z50',

    # Temperature (13 levels)
    't1000', 't925', 't850', 't700', 't600', 't500', 't400',
    't300', 't250', 't200', 't150', 't100', 't50',

    # U-component of wind (13 levels)
    'u1000', 'u925', 'u850', 'u700', 'u600', 'u500', 'u400',
    'u300', 'u250', 'u200', 'u150', 'u100', 'u50',

    # V-component of wind (13 levels)
    'v1000', 'v925', 'v850', 'v700', 'v600', 'v500', 'v400',
    'v300', 'v250', 'v200', 'v150', 'v100', 'v50',

    # Specific humidity (13 levels)
    'q1000', 'q925', 'q850', 'q700', 'q600', 'q500', 'q400',
    'q300', 'q250', 'q200', 'q150', 'q100', 'q50',

    # Surface variables (11 variables)
    't2m',   # 2m temperature
    'd2m',   # 2m dewpoint temperature
    'sst',   # Sea surface temperature
    'ttr',   # Top net thermal radiation
    '10u',   # 10m U-wind component
    '10v',   # 10m V-wind component
    '100u',  # 100m U-wind component
    '100v',  # 100m V-wind component
    'msl',   # Mean sea level pressure
    'tcwv',  # Total column water vapour
    'tp',    # Total precipitation
]
```

---

_Document generated: January 10, 2026_  
_FuXi-S2S Model Documentation v1.0_
