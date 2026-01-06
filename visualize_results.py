"""
Visualize FuXi-S2S vs PAGASA Comparison Results

Creates comprehensive plots to understand forecast accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_visualizations(result_dir='compare/result_improved', output_dir='compare/plots'):
    """Create all visualization plots."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find and load all CSV files
    csv_files = glob(os.path.join(result_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("❌ No comparison results found!")
        return
    
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Combine all results
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"⚠ Error loading {csv_file}: {e}")
    
    if not all_data:
        print("❌ No valid data found!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['valid_time'] = pd.to_datetime(combined['valid_time'])
    
    print(f"📊 Loaded {len(combined)} forecast points")
    print()
    
    # =========================================================================
    # PLOT 1: Rainfall Time Series Comparison
    # =========================================================================
    if 'RAINFALL' in combined.columns and 'tp' in combined.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort by date
        data = combined.sort_values('valid_time')
        
        ax.plot(data['valid_time'], data['RAINFALL'], 'o-', 
                label='PAGASA Observed', color='#2E86AB', linewidth=2, markersize=6)
        ax.plot(data['valid_time'], data['tp'], 's--', 
                label='FuXi-S2S Forecast', color='#A23B72', linewidth=2, markersize=5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rainfall (mm)', fontsize=12, fontweight='bold')
        ax.set_title('Rainfall: Observed vs Forecast', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '1_rainfall_timeseries.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 2: Temperature Time Series Comparison
    # =========================================================================
    if 't2m_celsius' in combined.columns and 'TMAX' in combined.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        data = combined.sort_values('valid_time')
        
        ax.plot(data['valid_time'], data['TMAX'], 'o-', 
                label='PAGASA Observed', color='#F18F01', linewidth=2, markersize=6)
        ax.plot(data['valid_time'], data['t2m_celsius'], 's--', 
                label='FuXi-S2S Forecast', color='#C73E1D', linewidth=2, markersize=5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Maximum Temperature: Observed vs Forecast', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '2_temperature_timeseries.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 3: Rainfall Scatter Plot (Observed vs Forecast)
    # =========================================================================
    if 'RAINFALL' in combined.columns and 'tp' in combined.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(combined['RAINFALL'], combined['tp'], 
                  alpha=0.6, s=80, c='#6A4C93', edgecolors='white', linewidth=1.5)
        
        # Perfect forecast line
        max_val = max(combined['RAINFALL'].max(), combined['tp'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Forecast', alpha=0.7)
        
        ax.set_xlabel('Observed Rainfall (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Forecast Rainfall (mm)', fontsize=12, fontweight='bold')
        ax.set_title('Rainfall: Observed vs Forecast Scatter', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add R² correlation
        correlation = np.corrcoef(combined['RAINFALL'].dropna(), combined['tp'].dropna())[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '3_rainfall_scatter.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 4: Temperature Scatter Plot (Observed vs Forecast)
    # =========================================================================
    if 't2m_celsius' in combined.columns and 'TMAX' in combined.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(combined['TMAX'], combined['t2m_celsius'], 
                  alpha=0.6, s=80, c='#FF6B35', edgecolors='white', linewidth=1.5)
        
        # Perfect forecast line
        min_val = min(combined['TMAX'].min(), combined['t2m_celsius'].min())
        max_val = max(combined['TMAX'].max(), combined['t2m_celsius'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Forecast', alpha=0.7)
        
        ax.set_xlabel('Observed Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Forecast Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Temperature: Observed vs Forecast Scatter', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add R² correlation
        correlation = np.corrcoef(combined['TMAX'].dropna(), combined['t2m_celsius'].dropna())[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '4_temperature_scatter.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 5: Error by Lead Time (RMSE)
    # =========================================================================
    if 'lead_time_days' in combined.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        lead_times = sorted(combined['lead_time_days'].unique())
        
        # Rainfall RMSE by lead time
        if 'precip_error' in combined.columns:
            rain_rmse = []
            for lt in lead_times:
                subset = combined[combined['lead_time_days'] == lt]
                rmse = np.sqrt((subset['precip_error']**2).mean())
                rain_rmse.append(rmse)
            
            ax1.plot(lead_times, rain_rmse, 'o-', linewidth=2.5, markersize=8, 
                    color='#2E86AB', label='Rainfall RMSE')
            ax1.axhline(y=5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (<5 mm)')
            ax1.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (<10 mm)')
            ax1.set_ylabel('RMSE (mm)', fontsize=12, fontweight='bold')
            ax1.set_title('Rainfall Forecast Error by Lead Time', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10, loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # Temperature RMSE by lead time
        if 'temp_error' in combined.columns:
            temp_rmse = []
            for lt in lead_times:
                subset = combined[combined['lead_time_days'] == lt]
                rmse = np.sqrt((subset['temp_error']**2).mean())
                temp_rmse.append(rmse)
            
            ax2.plot(lead_times, temp_rmse, 'o-', linewidth=2.5, markersize=8, 
                    color='#F18F01', label='Temperature RMSE')
            ax2.axhline(y=2, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (<2°C)')
            ax2.axhline(y=4, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (<4°C)')
            ax2.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('RMSE (°C)', fontsize=12, fontweight='bold')
            ax2.set_title('Temperature Forecast Error by Lead Time', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10, loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '5_error_by_leadtime.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 6: Error Distribution (Histogram)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'precip_error' in combined.columns:
        ax1.hist(combined['precip_error'].dropna(), bins=30, 
                color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Rainfall Error (mm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Rainfall Error Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
    
    if 'temp_error' in combined.columns:
        ax2.hist(combined['temp_error'].dropna(), bins=30, 
                color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax2.set_xlabel('Temperature Error (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Temperature Error Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, '6_error_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # =========================================================================
    # PLOT 7: Wind Speed Time Series Comparison
    # =========================================================================
    if 'wind_speed' in combined.columns and 'WINDSPEED ' in combined.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        data = combined.sort_values('valid_time')
        
        ax.plot(data['valid_time'], data['WINDSPEED '], 'o-', 
                label='PAGASA Observed', color='#008B8B', linewidth=2, markersize=6)
        ax.plot(data['valid_time'], data['wind_speed'], 's--', 
                label='FuXi-S2S Forecast', color='#20B2AA', linewidth=2, markersize=5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('Wind Speed: Observed vs Forecast', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '7_wind_speed_timeseries.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 8: Wind Direction Comparison (Circular Plot)
    # =========================================================================
    if 'wind_direction' in combined.columns and 'WIND DIRECTION' in combined.columns:
        # Filter out zero wind cases where direction is undefined
        mask = (combined['WIND DIRECTION'] > 0) & combined['wind_direction'].notna()
        wind_data = combined[mask].sort_values('valid_time')
        
        if len(wind_data) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(wind_data['valid_time'], wind_data['WIND DIRECTION'], 'o-', 
                    label='PAGASA Observed', color='#8B4513', linewidth=2, markersize=6)
            ax.plot(wind_data['valid_time'], wind_data['wind_direction'], 's--', 
                    label='FuXi-S2S Forecast', color='#D2691E', linewidth=2, markersize=5)
            
            # Add cardinal direction labels
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=180, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=270, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=360, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            
            # Add direction labels on right side
            ax.text(1.01, 0, 'N', transform=ax.get_yaxis_transform(), 
                   fontsize=10, va='center', color='gray')
            ax.text(1.01, 90/360, 'E', transform=ax.get_yaxis_transform(), 
                   fontsize=10, va='center', color='gray')
            ax.text(1.01, 180/360, 'S', transform=ax.get_yaxis_transform(), 
                   fontsize=10, va='center', color='gray')
            ax.text(1.01, 270/360, 'W', transform=ax.get_yaxis_transform(), 
                   fontsize=10, va='center', color='gray')
            ax.text(1.01, 1, 'N', transform=ax.get_yaxis_transform(), 
                   fontsize=10, va='center', color='gray')
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Wind Direction (°)', fontsize=12, fontweight='bold')
            ax.set_title('Wind Direction: Observed vs Forecast', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 360)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, '8_wind_direction_timeseries.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {plot_file}")
            plt.close()
    
    # =========================================================================
    # PLOT 9: Wind Speed Scatter Plot
    # =========================================================================
    if 'wind_speed' in combined.columns and 'WINDSPEED ' in combined.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        mask = combined['WINDSPEED '].notna() & combined['wind_speed'].notna()
        data = combined[mask]
        
        ax.scatter(data['WINDSPEED '], data['wind_speed'], 
                  alpha=0.6, s=80, c='#008B8B', edgecolors='white', linewidth=1.5)
        
        # Perfect forecast line
        max_val = max(data['WINDSPEED '].max(), data['wind_speed'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Forecast', alpha=0.7)
        
        ax.set_xlabel('Observed Wind Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Forecast Wind Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('Wind Speed: Observed vs Forecast Scatter', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add correlation
        if len(data) > 1:
            correlation = np.corrcoef(data['WINDSPEED '], data['wind_speed'])[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '9_wind_speed_scatter.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # PLOT 10: Wind Direction Polar Plot
    # =========================================================================
    if 'wind_direction' in combined.columns and 'WIND DIRECTION' in combined.columns:
        mask = (combined['WIND DIRECTION'] > 0) & combined['wind_direction'].notna()
        wind_data = combined[mask]
        
        if len(wind_data) > 0:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='polar')
            
            # Convert to radians (meteorological: 0° = North, clockwise)
            # Polar plot: 0° = East, counterclockwise, so we need to adjust
            obs_rad = np.deg2rad(90 - wind_data['WIND DIRECTION'])
            fcst_rad = np.deg2rad(90 - wind_data['wind_direction'])
            
            # Plot observed
            ax.scatter(obs_rad, np.ones(len(obs_rad))*1.0, 
                      s=100, alpha=0.6, c='#8B4513', label='Observed', edgecolors='white', linewidth=1.5)
            
            # Plot forecast
            ax.scatter(fcst_rad, np.ones(len(fcst_rad))*0.8, 
                      s=100, alpha=0.6, c='#D2691E', label='Forecast', marker='s', edgecolors='white', linewidth=1.5)
            
            # Configure polar plot
            ax.set_theta_zero_location('N')  # North at top
            ax.set_theta_direction(-1)  # Clockwise
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_title('Wind Direction Distribution (Polar View)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
            
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, '10_wind_direction_polar.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {plot_file}")
            plt.close()
    
    # =========================================================================
    # PLOT 11: Wind Error by Lead Time
    # =========================================================================
    if 'lead_time_days' in combined.columns and 'wind_speed' in combined.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        lead_times = sorted(combined['lead_time_days'].unique())
        
        # Wind Speed RMSE by lead time
        wind_speed_rmse = []
        for lt in lead_times:
            subset = combined[combined['lead_time_days'] == lt]
            mask = subset['WINDSPEED '].notna() & subset['wind_speed'].notna()
            if mask.sum() > 0:
                wind_err = subset.loc[mask, 'WINDSPEED '] - subset.loc[mask, 'wind_speed']
                rmse = np.sqrt((wind_err**2).mean())
                wind_speed_rmse.append(rmse)
            else:
                wind_speed_rmse.append(np.nan)
        
        valid_mask = ~np.isnan(wind_speed_rmse)
        if valid_mask.sum() > 0:
            ax1.plot(np.array(lead_times)[valid_mask], np.array(wind_speed_rmse)[valid_mask], 
                    'o-', linewidth=2.5, markersize=8, color='#008B8B', label='Wind Speed RMSE')
            ax1.axhline(y=1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (<1 m/s)')
            ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (<2 m/s)')
            ax1.set_ylabel('RMSE (m/s)', fontsize=12, fontweight='bold')
            ax1.set_title('Wind Speed Forecast Error by Lead Time', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10, loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # Wind Direction MAE by lead time
        if 'wind_dir_error' in combined.columns:
            wind_dir_mae = []
            for lt in lead_times:
                subset = combined[combined['lead_time_days'] == lt]
                mask = (subset['WIND DIRECTION'] > 0) & subset['wind_direction'].notna()
                if mask.sum() > 0:
                    mae = abs(subset.loc[mask, 'wind_dir_error']).mean()
                    wind_dir_mae.append(mae)
                else:
                    wind_dir_mae.append(np.nan)
            
            valid_mask = ~np.isnan(wind_dir_mae)
            if valid_mask.sum() > 0:
                ax2.plot(np.array(lead_times)[valid_mask], np.array(wind_dir_mae)[valid_mask], 
                        'o-', linewidth=2.5, markersize=8, color='#8B4513', label='Wind Direction MAE')
                ax2.axhline(y=45, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (<45°)')
                ax2.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair (<90°)')
                ax2.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('MAE (°)', fontsize=12, fontweight='bold')
                ax2.set_title('Wind Direction Forecast Error by Lead Time', fontsize=13, fontweight='bold')
                ax2.legend(fontsize=10, loc='upper left')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, '11_wind_error_by_leadtime.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 80)
    print(f"✅ ALL PLOTS SAVED TO: {output_dir}/")
    print("=" * 80)
    print()
    print("Generated plots:")
    print("  1.  1_rainfall_timeseries.png       - Rainfall over time")
    print("  2.  2_temperature_timeseries.png    - Temperature over time")
    print("  3.  3_rainfall_scatter.png          - Rainfall accuracy scatter")
    print("  4.  4_temperature_scatter.png       - Temperature accuracy scatter")
    print("  5.  5_error_by_leadtime.png         - Rain/Temp error vs lead time")
    print("  6.  6_error_distribution.png        - Distribution of Rain/Temp errors")
    print("  7.  7_wind_speed_timeseries.png     - Wind speed over time")
    print("  8.  8_wind_direction_timeseries.png - Wind direction over time")
    print("  9.  9_wind_speed_scatter.png        - Wind speed accuracy scatter")
    print("  10. 10_wind_direction_polar.png     - Wind direction polar distribution")
    print("  11. 11_wind_error_by_leadtime.png   - Wind error vs lead time")
    print()
    print("💡 Open the plots folder to view the visualizations!")
    print()


if __name__ == "__main__":
    create_visualizations()
