"""
Improved Comparison Script with Bias Correction and Ensemble Averaging

This script extends compare_pagasa.py with:
1. Ensemble averaging across all members
2. Bias correction using trained parameters
3. Improved metrics and reporting
"""

import argparse
import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime

# Import from existing utils
from utils.compare import (
    load_pagasa_excel, 
    load_fuxi_output, 
    extract_station_forecast,
    compare_forecasts
)
from utils.bias_correction import BiasCorrector, ensemble_average, load_training_data


def run_improved_comparison(
    fuxi_output='output',
    pagasa_file='data/PAGASA_CBSUA_Pili_2020_2023.xlsx',
    station_name='CBSUA Pili',
    station_lat=13.58,
    station_lon=123.28,
    num_members=5,
    init_date=None,
    apply_bias_correction=True,
    use_ensemble=True,
    output_dir='compare/result_improved'
):
    """
    Run improved comparison with bias correction and ensemble averaging.
    """
    
    print("=" * 80)
    print("IMPROVED FUXI-S2S vs PAGASA COMPARISON")
    print("=" * 80)
    print()
    print(f"Settings:")
    print(f"  Station: {station_name} ({station_lat}°N, {station_lon}°E)")
    print(f"  Ensemble members: {num_members}")
    print(f"  Bias correction: {'Enabled' if apply_bias_correction else 'Disabled'}")
    print(f"  Ensemble averaging: {'Enabled' if use_ensemble else 'Disabled'}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    station_dir = os.path.join(output_dir, station_name.replace(' ', '_'))
    os.makedirs(station_dir, exist_ok=True)
    
    # Load PAGASA data
    print("Loading PAGASA observations...")
    pagasa_data = load_pagasa_excel(pagasa_file)
    if pagasa_data is None:
        print("❌ Failed to load PAGASA data")
        return None
    
    # Find available forecast date
    if init_date is None:
        # Auto-detect latest forecast date
        year_dirs = sorted(glob(os.path.join(fuxi_output, '*')))
        if not year_dirs:
            print(f"❌ No forecast data found in {fuxi_output}")
            return None
        latest_year = year_dirs[-1]
        date_dirs = sorted(glob(os.path.join(latest_year, '*')))
        if not date_dirs:
            print(f"❌ No date directories found")
            return None
        init_date = os.path.basename(date_dirs[-1])
    
    print(f"Using init date: {init_date}")
    print()
    
    # =========================================================================
    # Load all ensemble members
    # =========================================================================
    print("Loading ensemble members...")
    member_forecasts = []
    
    for member in range(num_members):
        # Load and extract station forecast using existing function
        try:
            fuxi_data = load_fuxi_output(fuxi_output, member=member, init_date=init_date)
            if fuxi_data is None:
                print(f"  ⚠ Member {member:02d} not found, skipping")
                continue
                
            station_fcst = extract_station_forecast(
                fuxi_data, 
                station_lat, 
                station_lon,
                variables=['tp', 't2m', 'd2m', 'msl', '10u', '10v']
            )
            station_fcst['member'] = member
            member_forecasts.append(station_fcst)
            print(f"  ✓ Loaded member {member:02d} ({len(station_fcst)} time steps)")
        except Exception as e:
            print(f"  ⚠ Error loading member {member:02d}: {e}")
            continue
    
    if not member_forecasts:
        print("❌ No ensemble members loaded")
        return None
    
    print(f"\n✓ Loaded {len(member_forecasts)} ensemble members")
    print()
    
    # =========================================================================
    # Ensemble Averaging
    # =========================================================================
    if use_ensemble and len(member_forecasts) > 1:
        print("Computing ensemble mean...")
        ensemble_fcst = ensemble_average(member_forecasts)
        print(f"  ✓ Ensemble mean computed from {len(member_forecasts)} members")
        print()
    else:
        # Use first member
        ensemble_fcst = member_forecasts[0]
    
    # =========================================================================
    # Bias Correction
    # =========================================================================
    corrector = None
    if apply_bias_correction:
        print("Applying bias correction...")
        
        # Try to load existing parameters
        params_file = 'bias_correction_params.pkl'
        if os.path.exists(params_file):
            corrector = BiasCorrector()
            corrector.load(params_file)
            
            # Apply corrections
            ensemble_fcst = corrector.transform(ensemble_fcst)
            print("  ✓ Applied saved bias corrections")
        else:
            print("  ⚠ No bias correction parameters found")
            print("    Run 'python utils/bias_correction.py' first to train")
            apply_bias_correction = False
        print()
    
    # =========================================================================
    # Compare with PAGASA
    # =========================================================================
    print("Comparing with PAGASA observations...")
    
    # Merge forecast with observations
    ensemble_fcst['valid_time'] = pd.to_datetime(ensemble_fcst['valid_time'])
    pagasa_data['Date'] = pd.to_datetime(pagasa_data['Date'])
    
    # Convert temperature from Kelvin to Celsius if not already done
    if 't2m' in ensemble_fcst.columns and 't2m_celsius' not in ensemble_fcst.columns:
        ensemble_fcst['t2m_celsius'] = ensemble_fcst['t2m'] - 273.15
    
    comparison = pd.merge(
        pagasa_data,
        ensemble_fcst,
        left_on='Date',
        right_on='valid_time',
        how='inner'
    )
    
    if len(comparison) == 0:
        print("❌ No matching dates found")
        return None
    
    print(f"  ✓ Found {len(comparison)} matching dates")
    print()
    
    # =========================================================================
    # Calculate Errors (Raw and Corrected)
    # =========================================================================
    
    # Temperature - ensure t2m_celsius exists
    if 't2m' in comparison.columns and 't2m_celsius' not in comparison.columns:
        comparison['t2m_celsius'] = comparison['t2m'] - 273.15
    
    comparison['temp_error_raw'] = comparison['TMAX'] - comparison['t2m_celsius']
    if 't2m_corrected' in comparison.columns:
        comparison['temp_error_corrected'] = comparison['TMAX'] - comparison['t2m_corrected']
    
    # Rainfall
    comparison['precip_error_raw'] = comparison['RAINFALL'] - comparison['tp']
    if 'tp_corrected' in comparison.columns:
        comparison['precip_error_corrected'] = comparison['RAINFALL'] - comparison['tp_corrected']
    
    # Wind speed
    comparison['wind_error_raw'] = comparison['WINDSPEED '] - comparison['wind_speed']
    if 'wind_speed_corrected' in comparison.columns:
        comparison['wind_error_corrected'] = comparison['WINDSPEED '] - comparison['wind_speed_corrected']
    
    # =========================================================================
    # Print Comparison Results
    # =========================================================================
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Temperature
    temp_rmse_raw = np.sqrt((comparison['temp_error_raw'] ** 2).mean())
    temp_rmse_corr: float | None = None
    print(f"🌡️  TEMPERATURE (TMAX vs t2m)")
    print(f"   Raw RMSE:       {temp_rmse_raw:.2f} °C")
    if 't2m_corrected' in comparison.columns:
        temp_rmse_corr = np.sqrt((comparison['temp_error_corrected'] ** 2).mean())
        improvement = (temp_rmse_raw - temp_rmse_corr) / temp_rmse_raw * 100
        print(f"   Corrected RMSE: {temp_rmse_corr:.2f} °C ({improvement:+.1f}% improvement)")
    print()
    
    # Rainfall
    rain_rmse_raw = np.sqrt((comparison['precip_error_raw'] ** 2).mean())
    rain_rmse_corr: float | None = None
    print(f"🌧️  RAINFALL")
    print(f"   Raw RMSE:       {rain_rmse_raw:.2f} mm")
    if 'tp_corrected' in comparison.columns:
        rain_rmse_corr = np.sqrt((comparison['precip_error_corrected'] ** 2).mean())
        improvement = (rain_rmse_raw - rain_rmse_corr) / rain_rmse_raw * 100
        print(f"   Corrected RMSE: {rain_rmse_corr:.2f} mm ({improvement:+.1f}% improvement)")
    print()
    
    # Wind Speed
    wind_rmse_raw: float | None = None
    wind_rmse_corr: float | None = None
    mask = comparison['WINDSPEED '].notna() & comparison['wind_speed'].notna()
    if mask.sum() > 0:
        wind_rmse_raw = np.sqrt((comparison.loc[mask, 'wind_error_raw'] ** 2).mean())
        print(f"💨 WIND SPEED")
        print(f"   Raw RMSE:       {wind_rmse_raw:.2f} m/s")
        if 'wind_speed_corrected' in comparison.columns:
            wind_rmse_corr = np.sqrt((comparison.loc[mask, 'wind_error_corrected'] ** 2).mean())
            if wind_rmse_raw is not None and wind_rmse_corr is not None:
                improvement = (wind_rmse_raw - wind_rmse_corr) / wind_rmse_raw * 100
                print(f"   Corrected RMSE: {wind_rmse_corr:.2f} m/s ({improvement:+.1f}% improvement)")
        print()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("Saving results...")
    
    # Group by month and save - convert to period for grouping
    comparison['YearMonth'] = pd.to_datetime(comparison['valid_time']).dt.to_period('M')
    
    saved_files = []
    for period, group in comparison.groupby('YearMonth'):
        # Convert period to string for filename
        filename = f"{str(period).replace('-', '')[:6]}.csv"
        filepath = os.path.join(station_dir, filename)
        group.to_csv(filepath, index=False)
        saved_files.append(filepath)
        print(f"  📁 Saved: {filepath} ({len(group)} rows)")
    
    print()
    print(f"✅ Saved {len(saved_files)} monthly files to: {station_dir}")
    print()
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    summary_file = os.path.join(station_dir, 'SUMMARY.txt')
    with open(summary_file, 'w') as f:
        f.write("IMPROVED COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Station: {station_name}\n")
        f.write(f"Ensemble members: {len(member_forecasts)}\n")
        f.write(f"Bias correction: {'Applied' if apply_bias_correction else 'Not applied'}\n")
        f.write(f"Forecast points: {len(comparison)}\n\n")
        
        f.write("RAW FORECAST PERFORMANCE:\n")
        f.write(f"  Temperature RMSE: {temp_rmse_raw:.2f} °C\n")
        f.write(f"  Rainfall RMSE:    {rain_rmse_raw:.2f} mm\n")
        if wind_rmse_raw is not None:
            f.write(f"  Wind Speed RMSE:  {wind_rmse_raw:.2f} m/s\n")
        
        if apply_bias_correction:
            f.write("\nCORRECTED FORECAST PERFORMANCE:\n")
            if temp_rmse_corr is not None:
                f.write(f"  Temperature RMSE: {temp_rmse_corr:.2f} °C\n")
            if rain_rmse_corr is not None:
                f.write(f"  Rainfall RMSE:    {rain_rmse_corr:.2f} mm\n")
            if wind_rmse_corr is not None:
                f.write(f"  Wind Speed RMSE:  {wind_rmse_corr:.2f} m/s\n")
    
    print(f"📊 Summary saved to: {summary_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Improved FuXi-S2S comparison with bias correction')
    
    parser.add_argument('--fuxi_output', type=str, default='output',
                       help='FuXi output directory')
    parser.add_argument('--pagasa', type=str, default='data/PAGASA_CBSUA_Pili_2020_2023.xlsx',
                       help='PAGASA Excel file')
    parser.add_argument('--members', type=int, default=5,
                       help='Number of ensemble members to use')
    parser.add_argument('--init_date', type=str, default=None,
                       help='Initialization date (YYYYMMDD)')
    parser.add_argument('--no_correction', action='store_true',
                       help='Disable bias correction')
    parser.add_argument('--no_ensemble', action='store_true',
                       help='Disable ensemble averaging')
    parser.add_argument('--output_dir', type=str, default='compare/result_improved',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_improved_comparison(
        fuxi_output=args.fuxi_output,
        pagasa_file=args.pagasa,
        num_members=args.members,
        init_date=args.init_date,
        apply_bias_correction=not args.no_correction,
        use_ensemble=not args.no_ensemble,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()