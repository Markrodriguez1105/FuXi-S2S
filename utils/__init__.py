"""
FuXi-S2S Utility Package

Provides utility functions for:
- Display and logging
- Model loading
- Data processing and saving
- Inference execution
- PAGASA comparison
- Bias correction
"""

from .display import (
    print_header,
    print_footer,
    print_lite_mode_info,
    print_crop_mode_info,
    print_forecast_start_info,
    print_model_loading_info,
    print_completion_info,
)

from .model import load_model

from .data import (
    crop_to_region,
    save_forecast,
    save_with_progress,
    land_to_nan,
)

from .inference import run_inference

from .bias_correction import (
    BiasCorrector,
    ensemble_average,
    load_training_data,
    train_and_save_corrector,
)

from .compare import (
    STATIONS,
    VARIABLE_MAPPING,
    get_output_path,
    load_pagasa_excel,
    load_fuxi_output,
    extract_station_forecast,
    compare_forecasts,
    save_comparison_by_month,
    run_comparison,
)

__all__ = [
    # Display
    'print_header',
    'print_footer',
    'print_lite_mode_info',
    'print_crop_mode_info',
    'print_forecast_start_info',
    'print_model_loading_info',
    'print_completion_info',
    # Model
    'load_model',
    # Data
    'crop_to_region',
    'save_forecast',
    'save_with_progress',
    'land_to_nan',
    # Inference
    'run_inference',
    # Compare
    'STATIONS',
    'VARIABLE_MAPPING',
    'get_output_path',
    'load_pagasa_excel',
    'load_fuxi_output',
    'extract_station_forecast',
    'compare_forecasts',
    'save_comparison_by_month',
    'run_comparison',
]
