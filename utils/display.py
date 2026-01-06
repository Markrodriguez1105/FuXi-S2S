"""
Display and Logging Utilities

Functions for formatted console output and progress display.
"""


def print_header(title, width=70):
    """Print a formatted header."""
    print('\n' + '=' * width)
    print(title)
    print('=' * width)


def print_footer(width=70):
    """Print a footer line."""
    print('=' * width + '\n')


def print_lite_mode_info(total_step, total_member):
    """Display lightweight mode information."""
    print("=== LIGHTWEIGHT TESTING MODE ===")
    print(f"total_step: {total_step}, total_member: {total_member}")
    print("================================")


def print_crop_mode_info(crop_lat, crop_lon, crop_radius):
    """Display regional crop mode information."""
    print(f"\n=== REGIONAL OUTPUT MODE ===")
    print(f"⚠️  NOTE: Computation still uses global data (required by model)")
    print(f"📍 Center: ({crop_lat}°N, {crop_lon}°E)")
    print(f"📐 Radius: {crop_radius}° (~{crop_radius * 111:.0f} km)")
    lat_min = max(-90, crop_lat - crop_radius)
    lat_max = min(90, crop_lat + crop_radius)
    lon_min = crop_lon - crop_radius
    lon_max = crop_lon + crop_radius
    print(f"🗺️  Output region: {lat_min:.1f}° to {lat_max:.1f}°N, {lon_min:.1f}° to {lon_max:.1f}°E")
    print(f"💾 This reduces output file size significantly!")
    print("==============================")


def print_forecast_start_info(init_time, lat, lon, total_step, total_member):
    """Display forecast generation start information."""
    print_header('WEATHER FORECAST GENERATION STARTING')
    print(f'📅 Forecast Start Date: {init_time.strftime("%B %d, %Y at %H:00 UTC")}')
    print(f'🌍 Coverage Area: Latitude {lat[0]:.1f}° to {lat[-1]:.1f}°, Longitude {lon[0]:.1f}° to {lon[-1]:.1f}°')
    print(f'📊 Total Forecast Days: {total_step}')
    print(f'🔢 Ensemble Members: {total_member} (multiple scenarios for uncertainty)')
    print_footer()


def print_model_loading_info(device):
    """Display model loading information."""
    print_header('🤖 LOADING AI WEATHER FORECAST MODEL')
    print(f'Model: FuXi-S2S (Subseasonal-to-Seasonal Forecasting)')
    print(f'Device: {device.upper()}')


def print_completion_info(save_dir, total_member, total_step):
    """Display completion information."""
    print_header('🎉 FORECAST GENERATION COMPLETE!')
    if save_dir:
        print(f'📁 Forecast files saved to: {save_dir}')
        print(f'📊 Total forecasts generated: {total_member} ensemble members × {total_step} days')
    print_footer()
