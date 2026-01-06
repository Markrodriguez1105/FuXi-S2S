"""
Data Processing Utilities

Functions for data manipulation, cropping, and saving.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd


def crop_to_region(ds, crop_lat, crop_lon, crop_radius):
    """
    Crop DataArray to a regional bounding box.
    
    Parameters:
    -----------
    ds : xr.DataArray
        Data array to crop
    crop_lat : float
        Center latitude
    crop_lon : float
        Center longitude
    crop_radius : float
        Radius in degrees
    
    Returns:
    --------
    xr.DataArray
        Cropped data array
    """
    lat_min = max(-90, crop_lat - crop_radius)
    lat_max = min(90, crop_lat + crop_radius)
    lon_min = crop_lon - crop_radius
    lon_max = crop_lon + crop_radius
    
    # Handle longitude wrapping (0-360 range)
    if lon_min < 0:
        lon_min = lon_min + 360
    if lon_max > 360:
        lon_max = lon_max - 360
    
    # Select region (note: lat is 90 to -90, so max comes first)
    return ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    )


def save_forecast(output, input_data, member, lead_time, save_dir, 
                  crop_lat=None, crop_lon=None, crop_radius=10.0,
                  print_fn=None):
    """
    Save forecast output to NetCDF file.
    
    Parameters:
    -----------
    output : np.ndarray
        Model output array
    input_data : xr.DataArray
        Input data (for coordinates)
    member : int
        Ensemble member number
    lead_time : int
        Forecast lead time in days
    save_dir : str
        Base output directory
    crop_lat, crop_lon : float, optional
        Center coordinates for regional cropping
    crop_radius : float
        Radius in degrees for cropping
    print_fn : callable, optional
        Function to print data array info
    """
    if not save_dir:
        return
        
    init_time = pd.to_datetime(input_data.time.data[-1])
    
    # Organize by year and initialization date: output/YYYY/YYYYMMDD/member/XX/
    year_str = init_time.strftime("%Y")
    date_str = init_time.strftime("%Y%m%d")
    member_dir = os.path.join(save_dir, year_str, date_str, f"member/{member:02d}")
    os.makedirs(member_dir, exist_ok=True)

    ds = xr.DataArray(
        data=output,
        dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
        coords=dict(
            time=[init_time],
            lead_time=[lead_time],
            channel=input_data.channel,
            lat=input_data.lat,
            lon=input_data.lon,
        )
    ).astype(np.float32)
    
    # Crop to regional output if coordinates specified
    if crop_lat is not None and crop_lon is not None:
        ds = crop_to_region(ds, crop_lat, crop_lon, crop_radius)
    
    if print_fn:
        print_fn(ds)
    
    save_name = os.path.join(member_dir, f'{lead_time:02d}.nc')
    ds.to_netcdf(save_name)


def save_with_progress(ds, save_name, dtype=np.float32):
    """
    Save DataArray to NetCDF with progress bar.
    
    Parameters:
    -----------
    ds : xr.DataArray
        Data array to save
    save_name : str
        Output file path
    dtype : np.dtype
        Data type for output
    """
    from dask.diagnostics.progress import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)
    obj = ds.to_netcdf(save_name, compute=False)

    with ProgressBar():
        obj.compute()


def land_to_nan(input_data, mask, names=['sst']):
    """
    Set land values to NaN for specified variables.
    
    Parameters:
    -----------
    input_data : xr.DataArray
        Input data array
    mask : xr.DataArray
        Land/sea mask
    names : list
        Variable names to mask
    
    Returns:
    --------
    xr.DataArray
        Data with land values set to NaN
    """
    channel = input_data.channel.data.tolist()
    for ch in names:
        v = input_data.sel(channel=ch)
        v = v.where(mask)
        idx = channel.index(ch)
        input_data.data[:, idx] = v.data
    return input_data
