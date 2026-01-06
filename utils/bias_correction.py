"""
Bias Correction Module for FuXi-S2S Forecasts

Implements statistical post-processing to improve forecast accuracy:
1. Temperature bias correction (linear regression)
2. Rainfall calibration (scaling + threshold)
3. Wind corrections
4. Lead-time dependent adjustments
5. Ensemble averaging
"""

import pandas as pd
import numpy as np
from glob import glob
import os
import pickle


class BiasCorrector:
    """Statistical bias correction for FuXi-S2S forecasts."""
    
    def __init__(self):
        self.temp_bias = None
        self.temp_slope = None
        self.rain_scale = None
        self.rain_threshold = None
        self.wind_speed_bias = None
        self.lead_time_factors = {}
        self.is_fitted = False
    
    def fit(self, comparison_data):
        """
        Train bias correction parameters from historical comparisons.
        
        Parameters:
        -----------
        comparison_data : pd.DataFrame
            Combined comparison CSV data with observed and forecast columns
        """
        print("=" * 60)
        print("TRAINING BIAS CORRECTION")
        print("=" * 60)
        print(f"Training samples: {len(comparison_data)}")
        print()
        
        # =====================================================================
        # 1. Temperature Bias Correction
        # =====================================================================
        if 't2m_celsius' in comparison_data.columns and 'TMAX' in comparison_data.columns:
            mask = comparison_data['t2m_celsius'].notna() & comparison_data['TMAX'].notna()
            temp_fcst = comparison_data.loc[mask, 't2m_celsius'].values
            temp_obs = comparison_data.loc[mask, 'TMAX'].values
            
            # Simple linear regression: obs = slope * fcst + bias
            # Using numpy polyfit for simplicity (degree 1)
            coeffs = np.polyfit(temp_fcst, temp_obs, 1)
            self.temp_slope = coeffs[0]
            self.temp_bias = coeffs[1]
            
            # Calculate improvement
            raw_error = np.sqrt(((temp_obs - temp_fcst) ** 2).mean())
            corrected = self.temp_slope * temp_fcst + self.temp_bias
            new_error = np.sqrt(((temp_obs - corrected) ** 2).mean())
            
            print(f"🌡️  TEMPERATURE CORRECTION:")
            print(f"   Correction: T_corrected = {self.temp_slope:.3f} × T_forecast + {self.temp_bias:.2f}°C")
            print(f"   Raw RMSE: {raw_error:.2f}°C → Corrected RMSE: {new_error:.2f}°C")
            print(f"   Improvement: {(raw_error - new_error) / raw_error * 100:.1f}%")
            print()
        
        # =====================================================================
        # 2. Rainfall Calibration
        # =====================================================================
        if 'tp' in comparison_data.columns and 'RAINFALL' in comparison_data.columns:
            mask = comparison_data['tp'].notna() & comparison_data['RAINFALL'].notna()
            rain_fcst = comparison_data.loc[mask, 'tp'].values
            rain_obs = comparison_data.loc[mask, 'RAINFALL'].values
            
            # Find optimal threshold for rain detection
            # Test different thresholds and find best one
            best_threshold = 0.01
            best_score = float('inf')
            
            for threshold in np.arange(0.01, 0.5, 0.01):
                # Binary classification: did it rain?
                predicted_rain = rain_fcst > threshold
                observed_rain = rain_obs > 0.5  # Consider > 0.5mm as rain
                
                # Simple error: miss + false alarm
                misses = np.sum(observed_rain & ~predicted_rain)
                false_alarms = np.sum(~observed_rain & predicted_rain)
                score = misses + false_alarms
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            
            self.rain_threshold = best_threshold
            
            # Calculate scaling factor for non-zero forecasts
            # Use quantile matching instead of simple mean ratio
            fcst_nonzero = rain_fcst[rain_fcst > self.rain_threshold]
            obs_nonzero = rain_obs[rain_obs > 0.5]
            
            if len(fcst_nonzero) > 0 and len(obs_nonzero) > 0:
                # Match 75th percentile (captures typical rain intensity)
                fcst_p75 = np.percentile(fcst_nonzero, 75)
                obs_p75 = np.percentile(obs_nonzero, 75)
                
                if fcst_p75 > 0:
                    self.rain_scale = obs_p75 / fcst_p75
                else:
                    self.rain_scale = obs_nonzero.mean() / max(fcst_nonzero.mean(), 0.01)
            else:
                self.rain_scale = 1.0
            
            # Cap the scale factor to avoid extreme corrections
            self.rain_scale = min(self.rain_scale, 100.0)
            
            # Calculate improvement
            raw_error = np.sqrt(((rain_obs - rain_fcst) ** 2).mean())
            corrected = np.where(rain_fcst > self.rain_threshold, 
                                rain_fcst * self.rain_scale, 
                                0.0)
            new_error = np.sqrt(((rain_obs - corrected) ** 2).mean())
            
            print(f"🌧️  RAINFALL CORRECTION:")
            print(f"   Detection threshold: {self.rain_threshold:.3f} mm")
            print(f"   Scale factor: {self.rain_scale:.2f}×")
            print(f"   Formula: If forecast > {self.rain_threshold:.3f}, multiply by {self.rain_scale:.2f}")
            print(f"   Raw RMSE: {raw_error:.2f}mm → Corrected RMSE: {new_error:.2f}mm")
            print(f"   Improvement: {(raw_error - new_error) / raw_error * 100:.1f}%")
            print()
        
        # =====================================================================
        # 3. Wind Speed Correction
        # =====================================================================
        if 'wind_speed' in comparison_data.columns and 'WINDSPEED ' in comparison_data.columns:
            mask = comparison_data['wind_speed'].notna() & comparison_data['WINDSPEED '].notna()
            wind_fcst = comparison_data.loc[mask, 'wind_speed'].values
            wind_obs = comparison_data.loc[mask, 'WINDSPEED '].values
            
            # Simple bias (wind speed is already pretty good)
            self.wind_speed_bias = np.mean(wind_obs - wind_fcst)
            
            raw_error = np.sqrt(((wind_obs - wind_fcst) ** 2).mean())
            corrected = wind_fcst + self.wind_speed_bias
            new_error = np.sqrt(((wind_obs - corrected) ** 2).mean())
            
            print(f"💨 WIND SPEED CORRECTION:")
            print(f"   Bias adjustment: {self.wind_speed_bias:+.2f} m/s")
            print(f"   Raw RMSE: {raw_error:.2f}m/s → Corrected RMSE: {new_error:.2f}m/s")
            print(f"   Improvement: {(raw_error - new_error) / raw_error * 100:.1f}%")
            print()
        
        # =====================================================================
        # 4. Lead-Time Dependent Corrections
        # =====================================================================
        if 'lead_time_days' in comparison_data.columns:
            print(f"📈 LEAD-TIME DEPENDENT FACTORS:")
            
            for lead_time in sorted(comparison_data['lead_time_days'].unique()):
                subset = comparison_data[comparison_data['lead_time_days'] == lead_time]
                
                factors = {}
                
                # Temperature adjustment per lead time
                if 't2m_celsius' in subset.columns and 'TMAX' in subset.columns:
                    mask = subset['t2m_celsius'].notna() & subset['TMAX'].notna()
                    if mask.sum() > 0:
                        temp_bias_lt = (subset.loc[mask, 'TMAX'] - subset.loc[mask, 't2m_celsius']).mean()
                        factors['temp_bias'] = temp_bias_lt
                
                # Rainfall adjustment per lead time
                if 'tp' in subset.columns and 'RAINFALL' in subset.columns:
                    mask = subset['tp'].notna() & subset['RAINFALL'].notna()
                    if mask.sum() > 0 and subset.loc[mask, 'tp'].mean() > 0.01:
                        rain_ratio_lt = subset.loc[mask, 'RAINFALL'].mean() / max(subset.loc[mask, 'tp'].mean(), 0.01)
                        factors['rain_scale'] = min(rain_ratio_lt, 100.0)
                
                self.lead_time_factors[lead_time] = factors
            
            # Print sample of lead time factors
            sample_leads = [1, 7, 14, 21, 28, 35, 42]
            for lt in sample_leads:
                if lt in self.lead_time_factors:
                    factors = self.lead_time_factors[lt]
                    temp_adj = factors.get('temp_bias', 0)
                    rain_adj = factors.get('rain_scale', 1)
                    print(f"   Day {lt:2d}: Temp +{temp_adj:.1f}°C, Rain ×{rain_adj:.1f}")
            print()
        
        self.is_fitted = True
        print("=" * 60)
        print("✅ BIAS CORRECTION TRAINING COMPLETE")
        print("=" * 60)
        print()
    
    def transform(self, forecast_data, use_lead_time=True):
        """
        Apply bias corrections to forecast data.
        
        Parameters:
        -----------
        forecast_data : pd.DataFrame
            Forecast data with columns: tp, t2m_celsius, wind_speed, lead_time_days
        use_lead_time : bool
            Whether to apply lead-time specific corrections
            
        Returns:
        --------
        pd.DataFrame
            Corrected forecast data with new columns: 
            tp_corrected, t2m_corrected, wind_speed_corrected
        """
        if not self.is_fitted:
            raise ValueError("BiasCorrector not fitted. Call fit() first.")
        
        corrected = forecast_data.copy()
        
        # Convert temperature to Celsius if needed
        if 't2m' in corrected.columns and 't2m_celsius' not in corrected.columns:
            corrected['t2m_celsius'] = corrected['t2m'] - 273.15
        
        # Temperature correction
        if self.temp_slope is not None and 't2m_celsius' in corrected.columns:
            corrected['t2m_corrected'] = self.temp_slope * corrected['t2m_celsius'] + self.temp_bias
        
        # Rainfall correction
        if self.rain_scale is not None and 'tp' in corrected.columns:
            corrected['tp_corrected'] = np.where(
                corrected['tp'] > self.rain_threshold,
                corrected['tp'] * self.rain_scale,
                0.0
            )
        
        # Wind speed correction
        if self.wind_speed_bias is not None and 'wind_speed' in corrected.columns:
            corrected['wind_speed_corrected'] = corrected['wind_speed'] + self.wind_speed_bias
            corrected['wind_speed_corrected'] = corrected['wind_speed_corrected'].clip(lower=0)
        
        # Lead-time specific adjustments (optional refinement)
        if use_lead_time and 'lead_time_days' in corrected.columns:
            for idx, row in corrected.iterrows():
                lt = row['lead_time_days']
                if lt in self.lead_time_factors:
                    factors = self.lead_time_factors[lt]
                    
                    # Override with lead-time specific if available
                    if 'temp_bias' in factors and 't2m_celsius' in row:
                        corrected.loc[idx, 't2m_corrected'] = row['t2m_celsius'] + factors['temp_bias']
        
        return corrected
    
    def save(self, filepath='bias_correction_params.pkl'):
        """Save trained parameters to file."""
        params = {
            'temp_bias': self.temp_bias,
            'temp_slope': self.temp_slope,
            'rain_scale': self.rain_scale,
            'rain_threshold': self.rain_threshold,
            'wind_speed_bias': self.wind_speed_bias,
            'lead_time_factors': self.lead_time_factors,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"✅ Saved correction parameters to: {filepath}")
    
    def load(self, filepath='bias_correction_params.pkl'):
        """Load trained parameters from file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.temp_bias = params['temp_bias']
        self.temp_slope = params['temp_slope']
        self.rain_scale = params['rain_scale']
        self.rain_threshold = params['rain_threshold']
        self.wind_speed_bias = params['wind_speed_bias']
        self.lead_time_factors = params['lead_time_factors']
        self.is_fitted = params['is_fitted']
        print(f"✅ Loaded correction parameters from: {filepath}")


def ensemble_average(member_forecasts):
    """
    Calculate ensemble mean from multiple member forecasts.
    
    Parameters:
    -----------
    member_forecasts : list of pd.DataFrame
        List of forecast DataFrames, one per ensemble member
        
    Returns:
    --------
    pd.DataFrame
        Ensemble mean forecast with uncertainty (std)
    """
    if len(member_forecasts) == 0:
        raise ValueError("No member forecasts provided")
    
    if len(member_forecasts) == 1:
        return member_forecasts[0].copy()
    
    # Combine all members
    combined = pd.concat(member_forecasts, keys=range(len(member_forecasts)))
    
    # Numeric columns to average
    numeric_cols = ['tp', 't2m', 't2m_celsius', 'wind_speed', 'wind_direction', 
                    'd2m', 'msl', '10u', '10v']
    available_cols = [c for c in numeric_cols if c in combined.columns]
    
    # Group by valid_time and calculate mean/std
    ensemble_mean = combined.groupby('valid_time')[available_cols].mean()
    ensemble_std = combined.groupby('valid_time')[available_cols].std()
    
    # Add suffix to std columns
    ensemble_std.columns = [f'{c}_std' for c in ensemble_std.columns]
    
    # Merge mean and std
    result = pd.concat([ensemble_mean, ensemble_std], axis=1)
    
    # Add back non-numeric columns from first member
    first = member_forecasts[0].set_index('valid_time')
    for col in ['init_time', 'lead_time_days']:
        if col in first.columns:
            result[col] = first[col]
    
    result = result.reset_index()
    
    return result


def load_training_data(result_dir='compare/result'):
    """Load all comparison CSV files for training."""
    csv_files = glob(os.path.join(result_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"❌ No comparison CSV files found in {result_dir}")
        return None
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"⚠ Error loading {csv_file}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(combined)} training samples from {len(csv_files)} files")
    
    return combined


def train_and_save_corrector(result_dir='compare/result', output_file='bias_correction_params.pkl'):
    """Train bias corrector and save parameters."""
    
    # Load training data
    training_data = load_training_data(result_dir)
    
    if training_data is None:
        return None
    
    # Create and train corrector
    corrector = BiasCorrector()
    corrector.fit(training_data)
    
    # Save parameters
    corrector.save(output_file)
    
    return corrector


if __name__ == "__main__":
    # Train bias correction from existing comparison data
    print("\n" + "=" * 60)
    print("BIAS CORRECTION TRAINING")
    print("=" * 60 + "\n")
    
    corrector = train_and_save_corrector()
    
    if corrector is not None:
        print("\n" + "=" * 60)
        print("EXAMPLE: APPLYING CORRECTIONS")
        print("=" * 60 + "\n")
        
        # Load sample data and show corrected values
        training_data = load_training_data()
        
        if training_data is not None:
            # Apply corrections
            corrected = corrector.transform(training_data)
            
            # Show sample comparisons
            print("Sample Original vs Corrected values:")
            print("-" * 60)
            
            sample = corrected.head(5)
            
            if 't2m_celsius' in sample.columns and 't2m_corrected' in sample.columns:
                print("\nTemperature (first 5 forecasts):")
                print(f"  {'Observed':>10} {'Raw Fcst':>10} {'Corrected':>10}")
                for idx, row in sample.iterrows():
                    obs = row.get('TMAX', np.nan)
                    raw = row.get('t2m_celsius', np.nan)
                    corr = row.get('t2m_corrected', np.nan)
                    print(f"  {obs:>10.1f} {raw:>10.1f} {corr:>10.1f}")
            
            if 'tp' in sample.columns and 'tp_corrected' in sample.columns:
                print("\nRainfall (first 5 forecasts):")
                print(f"  {'Observed':>10} {'Raw Fcst':>10} {'Corrected':>10}")
                for idx, row in sample.iterrows():
                    obs = row.get('RAINFALL', np.nan)
                    raw = row.get('tp', np.nan)
                    corr = row.get('tp_corrected', np.nan)
                    print(f"  {obs:>10.2f} {raw:>10.2f} {corr:>10.2f}")
            
            # Calculate overall improvement
            print("\n" + "=" * 60)
            print("IMPROVEMENT SUMMARY")
            print("=" * 60)
            
            if 'TMAX' in corrected.columns and 't2m_celsius' in corrected.columns:
                raw_temp_rmse = np.sqrt(((corrected['TMAX'] - corrected['t2m_celsius']) ** 2).mean())
                corr_temp_rmse = np.sqrt(((corrected['TMAX'] - corrected['t2m_corrected']) ** 2).mean())
                print(f"\n🌡️  Temperature RMSE: {raw_temp_rmse:.2f}°C → {corr_temp_rmse:.2f}°C ({(raw_temp_rmse-corr_temp_rmse)/raw_temp_rmse*100:.1f}% better)")
            
            if 'RAINFALL' in corrected.columns and 'tp' in corrected.columns:
                raw_rain_rmse = np.sqrt(((corrected['RAINFALL'] - corrected['tp']) ** 2).mean())
                corr_rain_rmse = np.sqrt(((corrected['RAINFALL'] - corrected['tp_corrected']) ** 2).mean())
                print(f"🌧️  Rainfall RMSE: {raw_rain_rmse:.2f}mm → {corr_rain_rmse:.2f}mm ({(raw_rain_rmse-corr_rain_rmse)/raw_rain_rmse*100:.1f}% better)")
            
            if 'WINDSPEED ' in corrected.columns and 'wind_speed' in corrected.columns:
                mask = corrected['WINDSPEED '].notna() & corrected['wind_speed'].notna()
                raw_wind_rmse = np.sqrt(((corrected.loc[mask, 'WINDSPEED '] - corrected.loc[mask, 'wind_speed']) ** 2).mean())
                corr_wind_rmse = np.sqrt(((corrected.loc[mask, 'WINDSPEED '] - corrected.loc[mask, 'wind_speed_corrected']) ** 2).mean())
                print(f"💨 Wind Speed RMSE: {raw_wind_rmse:.2f}m/s → {corr_wind_rmse:.2f}m/s ({(raw_wind_rmse-corr_wind_rmse)/raw_wind_rmse*100:.1f}% better)")
            
            print("\n✅ Bias correction training complete!")
            print(f"   Parameters saved to: bias_correction_params.pkl")
            print()
