"""
PAGASA Comparison Utilities

Functions for comparing FuXi-S2S forecasts with PAGASA observations.
"""

import os
import glob
import pandas as pd
import xarray as xr
import numpy as np


# PAGASA station locations (add more as needed)
STATIONS = {
    'Pili, Camarines Sur': {'lat': 13.58, 'lon': 123.28},
    'Naga City': {'lat': 13.62, 'lon': 123.19},
    'Pacol, Naga City': {'lat': 13.657096, 'lon': 123.224535},
}

# Channel name mappings for FuXi-S2S output
VARIABLE_MAPPING = {
    'tp': 'total_precipitation',      # mm (already scaled by 1000 in preprocessing)
    't2m': '2m_temperature',          # Kelvin
    'd2m': '2m_dewpoint_temperature', # Kelvin
    'msl': 'mean_sea_level_pressure', # Pa
    '10u': '10m_u_wind',
    '10v': '10m_v_wind',
}


def get_output_path(base_dir, date, filename):
    """
    Generate output path organized by year/month.
    
    Parameters:
    -----------
    base_dir : str
        Base output directory (e.g., 'compare/result')
    date : pd.Timestamp or datetime
        Date for organizing output
    filename : str
        Output filename
    
    Returns:
    --------
    str
        Full path: base_dir/yyyy/mm/filename
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    year_str = date.strftime("%Y")
    month_str = date.strftime("%m")
    
    output_dir = os.path.join(base_dir, year_str, month_str)
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, filename)


def load_pagasa_excel(excel_path):
    """Load PAGASA weather data from Excel file."""
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return None
    
    print(f"PAGASA data columns: {df.columns.tolist()}")
    print(f"PAGASA data shape: {df.shape}")
    
    # Handle PAGASA format with separate YEAR, MONTH, DAY columns
    if 'YEAR' in df.columns and 'MONTH' in df.columns and 'DAY' in df.columns:
        df['Date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
        print(f"Created Date column from YEAR/MONTH/DAY")
    elif 'Date' not in df.columns:
        # Try to find a date-like column
        for col in df.columns:
            if 'date' in col.lower():
                df['Date'] = pd.to_datetime(df[col])
                break
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"First few rows:\n{df.head()}")
    
    return df


def load_fuxi_output(output_dir, member=0, init_date=None):
    """
    Load FuXi-S2S output from multiple lead time files.
    
    FuXi-S2S saves output as: output/YYYY/YYYYMMDD/member/XX/YY.nc
    where YYYY is year, YYYYMMDD is initialization date, XX is member number, YY is lead time
    
    Parameters:
    -----------
    output_dir : str
        Base output directory
    member : int
        Ensemble member number (0-10)
    init_date : str
        Initialization date in format 'YYYYMMDD' or 'YYYY/MM/DD' or 'YYYY-MM-DD'
    """
    # Handle different date formats
    # Structure: output/YYYY/YYYYMMDD/member/XX/
    if init_date:
        # Normalize date format to YYYYMMDD
        date_str = init_date.replace('/', '').replace('-', '')
        year_str = date_str[:4]
        member_dir = os.path.join(output_dir, year_str, date_str, f"member/{member:02d}")
    else:
        # Try to find available years and dates in output directory
        available_dates = []
        try:
            for year_dir in os.listdir(output_dir):
                year_path = os.path.join(output_dir, year_dir)
                if os.path.isdir(year_path) and year_dir.isdigit() and len(year_dir) == 4:
                    for date_dir in os.listdir(year_path):
                        date_path = os.path.join(year_path, date_dir)
                        if os.path.isdir(date_path) and date_dir.isdigit() and len(date_dir) == 8:
                            available_dates.append((year_dir, date_dir))
        except FileNotFoundError:
            pass
        
        if available_dates:
            print(f"Available forecast dates: {sorted([d[1] for d in available_dates])}")
            year_str, date_str = sorted(available_dates)[-1]  # Use latest date
            print(f"Using latest date: {year_str}/{date_str}")
            member_dir = os.path.join(output_dir, year_str, date_str, f"member/{member:02d}")
        else:
            # Fallback to old format without date
            member_dir = os.path.join(output_dir, f"member/{member:02d}")
    
    if not os.path.exists(member_dir):
        print(f"Member directory not found: {member_dir}")
        return None
    
    # Find all lead time files
    nc_files = sorted(glob.glob(os.path.join(member_dir, "*.nc")))
    
    if not nc_files:
        print(f"No .nc files found in {member_dir}")
        return None
    
    print(f"Found {len(nc_files)} forecast files")
    
    # Load and concatenate all lead times
    datasets = []
    for nc_file in nc_files:
        ds = xr.open_dataarray(nc_file)
        datasets.append(ds)
    
    # Concatenate along lead_time dimension
    combined = xr.concat(datasets, dim='lead_time')
    
    print(f"FuXi-S2S data shape: {combined.shape}")
    print(f"Channels: {combined.channel.values[:10]}...")  # First 10 channels
    print(f"Lat range: {combined.lat.values[0]:.2f} to {combined.lat.values[-1]:.2f}")
    print(f"Lon range: {combined.lon.values[0]:.2f} to {combined.lon.values[-1]:.2f}")
    
    return combined


def extract_station_forecast(fuxi_data, lat, lon, variables=['tp', 't2m']):
    """
    Extract forecast data for a specific station location.
    
    Parameters:
    -----------
    fuxi_data : xr.DataArray
        FuXi-S2S output with dims (time, lead_time, channel, lat, lon)
    lat : float
        Station latitude
    lon : float
        Station longitude
    variables : list
        List of channel names to extract
    
    Returns:
    --------
    pd.DataFrame with forecast values for each variable and lead time
    """
    # Select nearest grid point
    station_data = fuxi_data.sel(lat=lat, lon=lon, method='nearest')
    
    actual_lat = float(station_data.lat.values)
    actual_lon = float(station_data.lon.values)
    print(f"Selected grid point: ({actual_lat:.2f}, {actual_lon:.2f})")
    print(f"Distance from station: {abs(lat-actual_lat):.2f}° lat, {abs(lon-actual_lon):.2f}° lon")
    
    # Extract each variable
    results = []
    init_time = pd.to_datetime(station_data.time.values[0])
    
    for lead_time in station_data.lead_time.values:
        valid_time = init_time + pd.Timedelta(days=int(lead_time))
        row = {
            'init_time': init_time,
            'lead_time_days': int(lead_time),
            'valid_time': valid_time,
        }
        
        for var in variables:
            if var in station_data.channel.values:
                value = float(station_data.sel(lead_time=lead_time, channel=var).values)
                row[var] = value
            else:
                print(f"Warning: Variable '{var}' not found in channels")
        
        # Calculate wind speed and direction from u/v components if available
        if '10u' in station_data.channel.values and '10v' in station_data.channel.values:
            u10 = float(station_data.sel(lead_time=lead_time, channel='10u').values)
            v10 = float(station_data.sel(lead_time=lead_time, channel='10v').values)
            
            # Wind speed: sqrt(u^2 + v^2)
            wind_speed = np.sqrt(u10**2 + v10**2)
            row['wind_speed'] = wind_speed
            
            # Wind direction: atan2(u, v) converted to meteorological convention
            # (0° = from North, 90° = from East, etc.)
            wind_dir_rad = np.arctan2(u10, v10)
            wind_dir_deg = np.degrees(wind_dir_rad) + 180  # Convert to "from" direction
            wind_dir_deg = wind_dir_deg % 360  # Normalize to 0-360
            row['wind_direction'] = wind_dir_deg
        
        results.append(row)
    
    return pd.DataFrame(results)


def compare_forecasts(pagasa_df, fuxi_df, 
                      pagasa_date_col='Date',
                      pagasa_rain_col='RAINFALL',
                      pagasa_temp_col=None):
    """
    Compare PAGASA observations with FuXi-S2S forecasts.
    
    Returns comparison DataFrame with error metrics.
    """
    # Ensure date columns are datetime
    pagasa_df = pagasa_df.copy()
    pagasa_df[pagasa_date_col] = pd.to_datetime(pagasa_df[pagasa_date_col])
    
    # Merge on valid_time (FuXi) = Date (PAGASA)
    comparison = pd.merge(
        pagasa_df, 
        fuxi_df,
        left_on=pagasa_date_col,
        right_on='valid_time',
        how='inner'
    )
    
    if len(comparison) == 0:
        print("Warning: No overlapping dates found between datasets!")
        print(f"PAGASA date range: {pagasa_df[pagasa_date_col].min()} to {pagasa_df[pagasa_date_col].max()}")
        print(f"FuXi valid time range: {fuxi_df['valid_time'].min()} to {fuxi_df['valid_time'].max()}")
        return None
    
    print(f"\nFound {len(comparison)} matching dates for comparison")
    
    # Calculate precipitation error if available
    if pagasa_rain_col in comparison.columns and 'tp' in comparison.columns:
        # tp from FuXi-S2S is already in mm (scaled in preprocessing)
        comparison['precip_error'] = comparison[pagasa_rain_col] - comparison['tp']
        comparison['precip_abs_error'] = abs(comparison['precip_error'])
        
        rmse = np.sqrt((comparison['precip_error']**2).mean())
        mae = comparison['precip_abs_error'].mean()
        bias = comparison['precip_error'].mean()
        
        print(f"\n=== Precipitation Comparison ===")
        print(f"RMSE: {rmse:.2f} mm")
        print(f"MAE:  {mae:.2f} mm")
        print(f"Bias: {bias:.2f} mm (positive = model underestimates)")
    
    # Calculate temperature error if available (comparing with TMAX)
    if pagasa_temp_col and pagasa_temp_col in comparison.columns and 't2m' in comparison.columns:
        # Convert FuXi t2m from Kelvin to Celsius
        comparison['t2m_celsius'] = comparison['t2m'] - 273.15
        comparison['temp_error'] = comparison[pagasa_temp_col] - comparison['t2m_celsius']
        
        temp_rmse = np.sqrt((comparison['temp_error']**2).mean())
        temp_mae = abs(comparison['temp_error']).mean()
        temp_bias = comparison['temp_error'].mean()
        
        print(f"\n=== Temperature Comparison ({pagasa_temp_col} vs t2m) ===")
        print(f"RMSE: {temp_rmse:.2f} °C")
        print(f"MAE:  {temp_mae:.2f} °C")
        print(f"Bias: {temp_bias:.2f} °C")
    
    # Also compare with TMIN if available
    if 'TMIN' in comparison.columns and 't2m' in comparison.columns:
        comparison['tmin_error'] = comparison['TMIN'] - comparison['t2m_celsius']
        tmin_rmse = np.sqrt((comparison['tmin_error']**2).mean())
        print(f"\n=== TMIN vs t2m Comparison ===")
        print(f"RMSE: {tmin_rmse:.2f} °C")
    
    # Wind comparison if available
    if 'WINDSPEED' in comparison.columns and 'wind_speed' in comparison.columns:
        comparison['wind_speed_error'] = comparison['WINDSPEED'] - comparison['wind_speed']
        comparison['wind_speed_abs_error'] = abs(comparison['wind_speed_error'])
        
        wind_rmse = np.sqrt((comparison['wind_speed_error']**2).mean())
        wind_mae = comparison['wind_speed_abs_error'].mean()
        wind_bias = comparison['wind_speed_error'].mean()
        
        print(f"\n=== Wind Speed Comparison ===")
        print(f"RMSE: {wind_rmse:.2f} m/s")
        print(f"MAE:  {wind_mae:.2f} m/s")
        print(f"Bias: {wind_bias:.2f} m/s")
    
    # Wind direction comparison if available
    if 'WIND DIRECTION' in comparison.columns and 'wind_direction' in comparison.columns:
        # Calculate circular difference for wind direction
        obs_dir = comparison['WIND DIRECTION'].values
        fcst_dir = comparison['wind_direction'].values
        
        # Circular difference (handles wrapping at 0/360)
        dir_diff = np.abs(obs_dir - fcst_dir)
        dir_diff = np.minimum(dir_diff, 360 - dir_diff)
        comparison['wind_dir_error'] = dir_diff
        
        dir_mae = dir_diff.mean()
        
        print(f"\n=== Wind Direction Comparison ===")
        print(f"MAE:  {dir_mae:.1f}° (circular difference)")
    
    return comparison


def save_comparison_by_month(comparison_df, location, base_output_dir='compare/result', 
                             date_col='valid_time'):
    """
    Save comparison results organized by location with monthly files.
    
    Output structure: compare/result/[location]/yyyymm.csv
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results DataFrame
    location : str
        Location/station name for folder
    base_output_dir : str
        Base directory for output (default: 'compare/result')
    date_col : str
        Column containing dates for organizing
    
    Returns:
    --------
    list
        List of saved file paths
    """
    if comparison_df is None or len(comparison_df) == 0:
        print("No data to save")
        return []
    
    saved_files = []
    
    # Sanitize location name for folder
    location_folder = location.replace(' ', '_').replace(',', '').replace('/', '_')
    
    # Group by year-month
    comparison_df = comparison_df.copy()
    comparison_df['_year'] = pd.to_datetime(comparison_df[date_col]).dt.year
    comparison_df['_month'] = pd.to_datetime(comparison_df[date_col]).dt.month
    
    for (year, month), group in comparison_df.groupby(['_year', '_month']):
        # Remove helper columns
        group = group.drop(columns=['_year', '_month'])
        
        # Create output path: compare/result/[location]/yyyymm.csv
        output_dir = os.path.join(base_output_dir, location_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{year}{month:02d}.csv"
        output_path = os.path.join(output_dir, filename)
        
        group.to_csv(output_path, index=False)
        saved_files.append(output_path)
        print(f"  📁 Saved: {output_path} ({len(group)} rows)")
    
    return saved_files


def run_comparison(pagasa_path, fuxi_output_dir, station='CBSUA Pili', 
                   member=0, init_date=None, lat=None, lon=None,
                   output_dir='compare/result'):
    """
    Run full comparison between FuXi-S2S and PAGASA data.
    
    Parameters:
    -----------
    pagasa_path : str
        Path to PAGASA Excel file
    fuxi_output_dir : str
        Path to FuXi-S2S output directory
    station : str
        Station name for coordinates
    member : int
        Ensemble member to use (0-10)
    init_date : str
        Initialization date (YYYYMMDD)
    lat, lon : float
        Override station coordinates
    output_dir : str
        Base output directory (default: compare/result)
    
    Returns:
    --------
    pd.DataFrame or None
        Comparison results
    """
    # Get station coordinates
    if lat is not None and lon is not None:
        station_lat, station_lon = lat, lon
    elif station in STATIONS:
        station_lat = STATIONS[station]['lat']
        station_lon = STATIONS[station]['lon']
    else:
        print(f"Unknown station: {station}")
        print(f"Available stations: {list(STATIONS.keys())}")
        print("Please specify lat and lon manually")
        return None
    
    print(f"=== FuXi-S2S vs PAGASA Comparison ===")
    print(f"Station: {station} ({station_lat}°N, {station_lon}°E)")
    print(f"Output directory: {output_dir}/{station}/")
    print()
    
    # Load PAGASA data
    print("Loading PAGASA data...")
    pagasa_df = load_pagasa_excel(pagasa_path)
    if pagasa_df is None:
        return None
    
    print()
    
    # Load FuXi-S2S output
    print("Loading FuXi-S2S forecast data...")
    fuxi_data = load_fuxi_output(fuxi_output_dir, member=member, init_date=init_date)
    if fuxi_data is None:
        print("\nPlease run inference.py first to generate FuXi-S2S forecasts.")
        return None
    
    print()
    
    # Extract station forecast
    print("Extracting station forecast...")
    fuxi_df = extract_station_forecast(
        fuxi_data, station_lat, station_lon, 
        variables=['tp', 't2m', 'd2m', 'msl', '10u', '10v']
    )
    print(f"Forecast DataFrame shape: {fuxi_df.shape}")
    print(fuxi_df.head())
    
    print()
    
    # Compare datasets
    print("Comparing datasets...")
    comparison = compare_forecasts(
        pagasa_df, 
        fuxi_df,
        pagasa_date_col='Date',
        pagasa_rain_col='RAINFALL',
        pagasa_temp_col='TMAX'
    )
    
    if comparison is not None:
        # Save results organized by location with monthly files
        print(f"\n📁 Saving results to {output_dir}/{station}/...")
        saved_files = save_comparison_by_month(
            comparison,
            location=station,
            base_output_dir=output_dir,
            date_col='valid_time'
        )
        
        print(f"\n✅ Saved {len(saved_files)} monthly files")
        
        # Show sample comparison
        print("\nSample comparison rows:")
        cols = ['valid_time', 'lead_time_days', 'RAINFALL', 'tp', 'precip_error']
        available_cols = [c for c in cols if c in comparison.columns]
        print(comparison[available_cols].head(10))
    
    return comparison
