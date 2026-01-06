"""
Inference Utilities

Main inference loop for running FuXi-S2S forecasts.
"""

import time
import numpy as np
import pandas as pd
from copy import deepcopy

from .display import print_forecast_start_info
from .data import save_forecast


def run_inference(model, input_data, total_step, total_member, input_names,
                  save_dir="", crop_lat=None, crop_lon=None, crop_radius=10.0,
                  print_fn=None):
    """
    Run FuXi-S2S inference for multiple ensemble members.
    
    Parameters:
    -----------
    model : ort.InferenceSession
        Loaded ONNX model
    input_data : xr.DataArray
        Input data array
    total_step : int
        Number of forecast days
    total_member : int
        Number of ensemble members
    input_names : list
        Model input names
    save_dir : str
        Output directory
    crop_lat, crop_lon : float, optional
        Center coordinates for regional output
    crop_radius : float
        Radius for regional cropping
    print_fn : callable, optional
        Function to print data array info
    """
    hist_time = pd.to_datetime(input_data.time.values[-2])
    init_time = pd.to_datetime(input_data.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(days=1)
    
    lat = input_data.lat.values
    lon = input_data.lon.values
    batch = input_data.values[None]
    
    assert lat[0] == 90 and lat[-1] == -90
    print_forecast_start_info(init_time, lat, lon, total_step, total_member)

    for member in range(total_member):
        print(f'\n🎲 ENSEMBLE MEMBER {member+1}/{total_member}')
        print(f'   Generating forecast scenario #{member+1}...')
        new_input = deepcopy(batch)

        start = time.perf_counter()
        for step in range(total_step):
            lead_time = (step + 1)

            inputs = {'input': new_input}

            if "step" in input_names:
                inputs['step'] = np.array([step], dtype=np.float32)

            if "doy" in input_names:
                valid_time = init_time + pd.Timedelta(days=step)
                doy = min(365, valid_time.day_of_year) / 365
                inputs['doy'] = np.array([doy], dtype=np.float32)

            istart = time.perf_counter()
            new_input, = model.run(None, inputs)
            output = deepcopy(new_input[:, -1:])
            step_time = time.perf_counter() - istart

            forecast_date = init_time + pd.Timedelta(days=lead_time)
            print(f"   ✓ Day {lead_time:2d} ({forecast_date.strftime('%b %d, %Y')}) - Computed in {step_time:.2f}s")
            
            save_forecast(
                output, input_data, member, lead_time, save_dir,
                crop_lat=crop_lat, crop_lon=crop_lon, crop_radius=crop_radius,
                print_fn=print_fn
            )
            
            if step > total_step:
                break

        run_time = time.perf_counter() - start
        print(f'   ✅ Ensemble member {member+1} complete! Total time: {run_time:.1f}s ({run_time/60:.1f} minutes)')
