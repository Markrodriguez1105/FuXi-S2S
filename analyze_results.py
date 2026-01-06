"""
Analyze FuXi-S2S vs PAGASA Comparison Results

This script reads comparison CSV files and generates an easy-to-understand summary report.
"""

import pandas as pd
import numpy as np
import os
from glob import glob

def analyze_comparison_results(result_dir='compare/result'):
    """Analyze all comparison CSV files and generate summary."""
    
    # Find all CSV files
    csv_files = glob(os.path.join(result_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("❌ No comparison results found!")
        print(f"   Expected location: {result_dir}/[station]/YYYYMM.csv")
        return
    
    print("=" * 80)
    print("FUXI-S2S FORECAST ACCURACY ANALYSIS")
    print("=" * 80)
    print()
    
    # Combine all results
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
            station = os.path.basename(os.path.dirname(csv_file))
            month = os.path.basename(csv_file).replace('.csv', '')
            print(f"✓ Loaded: {station} - {month} ({len(df)} forecasts)")
        except Exception as e:
            print(f"⚠ Error loading {csv_file}: {e}")
    
    if not all_data:
        print("❌ No valid data found!")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    print()
    print("=" * 80)
    print(f"SUMMARY STATISTICS ({len(combined)} total forecast points)")
    print("=" * 80)
    print()
    
    # Overall statistics
    print("📊 OVERALL FORECAST PERFORMANCE")
    print("-" * 80)
    
    # Precipitation analysis
    if 'precip_error' in combined.columns and 'RAINFALL' in combined.columns:
        rain_obs = combined['RAINFALL'].dropna()
        rain_fcst = combined['tp'].dropna()
        rain_error = combined['precip_error'].dropna()
        
        rmse = np.sqrt((rain_error**2).mean())
        mae = abs(rain_error).mean()
        bias = rain_error.mean()
        
        print(f"\n🌧️  RAINFALL (Precipitation)")
        print(f"   Observed Range:     {rain_obs.min():.1f} - {rain_obs.max():.1f} mm")
        print(f"   Forecast Range:     {rain_fcst.min():.1f} - {rain_fcst.max():.1f} mm")
        print(f"   Average Observed:   {rain_obs.mean():.2f} mm")
        print(f"   Average Forecast:   {rain_fcst.mean():.2f} mm")
        print(f"   ")
        print(f"   RMSE (Error):       {rmse:.2f} mm")
        print(f"   MAE (Avg Error):    {mae:.2f} mm")
        print(f"   Bias:               {bias:.2f} mm", end="")
        if abs(bias) < 0.5:
            print(" ✓ (Nearly unbiased)")
        elif bias > 0:
            print(" ⚠ (Model tends to underestimate)")
        else:
            print(" ⚠ (Model tends to overestimate)")
    
    # Temperature analysis
    if 't2m_celsius' in combined.columns and 'TMAX' in combined.columns:
        temp_obs = combined['TMAX'].dropna()
        temp_fcst = combined['t2m_celsius'].dropna()
        temp_error = combined['temp_error'].dropna()
        
        temp_rmse = np.sqrt((temp_error**2).mean())
        temp_mae = abs(temp_error).mean()
        temp_bias = temp_error.mean()
        
        print(f"\n🌡️  TEMPERATURE (TMAX vs 2m Temperature)")
        print(f"   Observed Range:     {temp_obs.min():.1f} - {temp_obs.max():.1f} °C")
        print(f"   Forecast Range:     {temp_fcst.min():.1f} - {temp_fcst.max():.1f} °C")
        print(f"   Average Observed:   {temp_obs.mean():.1f} °C")
        print(f"   Average Forecast:   {temp_fcst.mean():.1f} °C")
        print(f"   ")
        print(f"   RMSE (Error):       {temp_rmse:.2f} °C")
        print(f"   MAE (Avg Error):    {temp_mae:.2f} °C")
        print(f"   Bias:               {temp_bias:.2f} °C", end="")
        if abs(temp_bias) < 0.5:
            print(" ✓ (Nearly unbiased)")
        elif temp_bias > 0:
            print(" ⚠ (Model tends to underestimate)")
        else:
            print(" ⚠ (Model tends to overestimate)")
    
    # Wind Speed analysis
    if 'wind_speed' in combined.columns and 'WINDSPEED ' in combined.columns:
        # Note: PAGASA column has trailing space
        wind_obs = combined['WINDSPEED '].dropna()
        wind_fcst = combined['wind_speed'].dropna()
        
        # Calculate matching pairs
        mask = combined['WINDSPEED '].notna() & combined['wind_speed'].notna()
        wind_error = combined.loc[mask, 'WINDSPEED '] - combined.loc[mask, 'wind_speed']
        
        if len(wind_error) > 0:
            wind_rmse = np.sqrt((wind_error**2).mean())
            wind_mae = abs(wind_error).mean()
            wind_bias = wind_error.mean()
            
            print(f"\n💨 WIND SPEED (10m)")
            print(f"   Observed Range:     {wind_obs.min():.1f} - {wind_obs.max():.1f} m/s")
            print(f"   Forecast Range:     {wind_fcst.min():.1f} - {wind_fcst.max():.1f} m/s")
            print(f"   Average Observed:   {wind_obs.mean():.2f} m/s")
            print(f"   Average Forecast:   {wind_fcst.mean():.2f} m/s")
            print(f"   ")
            print(f"   RMSE (Error):       {wind_rmse:.2f} m/s")
            print(f"   MAE (Avg Error):    {wind_mae:.2f} m/s")
            print(f"   Bias:               {wind_bias:.2f} m/s", end="")
            if abs(wind_bias) < 0.5:
                print(" ✓ (Nearly unbiased)")
            elif wind_bias > 0:
                print(" ⚠ (Model tends to underestimate)")
            else:
                print(" ⚠ (Model tends to overestimate)")
    
    # Wind Direction analysis (circular statistics)
    if 'wind_dir_error' in combined.columns and 'WIND DIRECTION' in combined.columns:
        # Filter out zero wind observations (direction undefined)
        mask = (combined['WIND DIRECTION'] > 0) & combined['wind_direction'].notna()
        if mask.sum() > 0:
            dir_obs = combined.loc[mask, 'WIND DIRECTION']
            dir_fcst = combined.loc[mask, 'wind_direction']
            
            # Circular MAE (already calculated in wind_dir_error)
            dir_error = combined.loc[mask, 'wind_dir_error']
            dir_mae = abs(dir_error).mean()
            
            # Calculate correlation using circular correlation
            # Convert to radians for circular statistics
            obs_rad = np.deg2rad(dir_obs)
            fcst_rad = np.deg2rad(dir_fcst)
            
            print(f"\n🧭 WIND DIRECTION")
            print(f"   Valid Observations: {mask.sum()} (non-zero wind)")
            print(f"   Observed Range:     {dir_obs.min():.0f}° - {dir_obs.max():.0f}°")
            print(f"   Forecast Range:     {dir_fcst.min():.0f}° - {dir_fcst.max():.0f}°")
            print(f"   ")
            print(f"   MAE (Circular):     {dir_mae:.1f}°", end="")
            if dir_mae < 45:
                print(" ✓ (Excellent)")
            elif dir_mae < 90:
                print(" ⚠ (Fair - within quadrant)")
            else:
                print(" ⚠ (Poor - large directional errors)")
    
    # Lead time analysis
    if 'lead_time_days' in combined.columns:
        print()
        print("=" * 80)
        print("📈 FORECAST ACCURACY BY LEAD TIME")
        print("=" * 80)
        print()
        
        for lead_time in sorted(combined['lead_time_days'].unique()):
            subset = combined[combined['lead_time_days'] == lead_time]
            
            metrics = []
            
            if 'precip_error' in subset.columns:
                rain_rmse = np.sqrt((subset['precip_error']**2).mean())
                metrics.append(f"Rain RMSE = {rain_rmse:5.2f} mm")
                
            if 'temp_error' in subset.columns:
                temp_rmse = np.sqrt((subset['temp_error']**2).mean())
                metrics.append(f"Temp RMSE = {temp_rmse:4.2f} °C")
            
            # Wind speed RMSE
            if 'wind_speed' in subset.columns and 'WINDSPEED ' in subset.columns:
                mask = subset['WINDSPEED '].notna() & subset['wind_speed'].notna()
                if mask.sum() > 0:
                    wind_err = subset.loc[mask, 'WINDSPEED '] - subset.loc[mask, 'wind_speed']
                    wind_rmse = np.sqrt((wind_err**2).mean())
                    metrics.append(f"Wind RMSE = {wind_rmse:4.2f} m/s")
            
            # Wind direction MAE
            if 'wind_dir_error' in subset.columns:
                mask = (subset['WIND DIRECTION'] > 0) & subset['wind_direction'].notna()
                if mask.sum() > 0:
                    dir_mae = abs(subset.loc[mask, 'wind_dir_error']).mean()
                    metrics.append(f"Wind Dir MAE = {dir_mae:5.1f}°")
            
            if metrics:
                print(f"Day {lead_time:2d}:  " + "  |  ".join(metrics))
            else:
                print(f"Day {lead_time:2d}:  No data")
    
    # Best and worst forecasts
    print()
    print("=" * 80)
    print("🎯 BEST & WORST FORECASTS")
    print("=" * 80)
    print()
    
    if 'precip_abs_error' in combined.columns and len(combined) > 0:
        # Best rainfall forecast
        best_rain = combined.loc[combined['precip_abs_error'].idxmin()]
        print(f"Best Rainfall Forecast:")
        print(f"   Date: {best_rain.get('valid_time', 'N/A')}")
        print(f"   Observed: {best_rain.get('RAINFALL', 0):.2f} mm")
        print(f"   Forecast: {best_rain.get('tp', 0):.2f} mm")
        print(f"   Error: {best_rain.get('precip_error', 0):.2f} mm")
        print()
        
        # Worst rainfall forecast
        worst_rain = combined.loc[combined['precip_abs_error'].idxmax()]
        print(f"Worst Rainfall Forecast:")
        print(f"   Date: {worst_rain.get('valid_time', 'N/A')}")
        print(f"   Observed: {worst_rain.get('RAINFALL', 0):.2f} mm")
        print(f"   Forecast: {worst_rain.get('tp', 0):.2f} mm")
        print(f"   Error: {worst_rain.get('precip_error', 0):.2f} mm")
    
    # Interpretation guide
    print()
    print("=" * 80)
    print("📖 INTERPRETATION GUIDE")
    print("=" * 80)
    print()
    print("RMSE (Root Mean Square Error):")
    print("  - Lower is better (0 = perfect)")
    print("  - Rainfall:      < 5 mm = Excellent, 5-10 mm = Good, > 10 mm = Fair")
    print("  - Temperature:   < 2°C = Excellent, 2-4°C = Good, > 4°C = Fair")
    print("  - Wind Speed:    < 1 m/s = Excellent, 1-2 m/s = Good, > 2 m/s = Fair")
    print()
    print("MAE (Mean Absolute Error):")
    print("  - Average size of errors (easier to interpret than RMSE)")
    print("  - Wind Direction: < 45° = Excellent, 45-90° = Fair, > 90° = Poor")
    print()
    print("Bias:")
    print("  - Positive = Model underestimates (forecasts too low)")
    print("  - Negative = Model overestimates (forecasts too high)")
    print("  - Close to 0 = Unbiased (good!)")
    print()
    print("Wind Direction (Circular Statistics):")
    print("  - Uses circular difference (accounts for 0°/360° wrapping)")
    print("  - 0° = from North, 90° = from East, 180° = from South, 270° = from West")
    print()
    print("=" * 80)
    
    # Save summary report
    summary_file = os.path.join(result_dir, 'ANALYSIS_SUMMARY.txt')
    with open(summary_file, 'w') as f:
        f.write("FuXi-S2S Forecast Accuracy Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total forecast points: {len(combined)}\n")
        
        if 'precip_error' in combined.columns:
            f.write(f"\nRAINFALL:\n")
            f.write(f"  RMSE: {rmse:.2f} mm\n")
            f.write(f"  MAE:  {mae:.2f} mm\n")
            f.write(f"  Bias: {bias:.2f} mm\n")
        
        if 't2m_celsius' in combined.columns:
            f.write(f"\nTEMPERATURE:\n")
            f.write(f"  RMSE: {temp_rmse:.2f} °C\n")
            f.write(f"  MAE:  {temp_mae:.2f} °C\n")
            f.write(f"  Bias: {temp_bias:.2f} °C\n")
        
        # Add wind metrics to summary
        if 'wind_speed' in combined.columns and 'WINDSPEED ' in combined.columns:
            mask = combined['WINDSPEED '].notna() & combined['wind_speed'].notna()
            if mask.sum() > 0:
                wind_error = combined.loc[mask, 'WINDSPEED '] - combined.loc[mask, 'wind_speed']
                wind_rmse = np.sqrt((wind_error**2).mean())
                wind_mae = abs(wind_error).mean()
                wind_bias = wind_error.mean()
                
                f.write(f"\nWIND SPEED:\n")
                f.write(f"  RMSE: {wind_rmse:.2f} m/s\n")
                f.write(f"  MAE:  {wind_mae:.2f} m/s\n")
                f.write(f"  Bias: {wind_bias:.2f} m/s\n")
        
        if 'wind_dir_error' in combined.columns:
            mask = (combined['WIND DIRECTION'] > 0) & combined['wind_direction'].notna()
            if mask.sum() > 0:
                dir_mae = abs(combined.loc[mask, 'wind_dir_error']).mean()
                f.write(f"\nWIND DIRECTION:\n")
                f.write(f"  MAE (Circular): {dir_mae:.1f}°\n")
                f.write(f"  Valid observations: {mask.sum()}\n")
    
    print(f"\n✅ Summary saved to: {summary_file}")
    print()


if __name__ == "__main__":
    analyze_comparison_results()
