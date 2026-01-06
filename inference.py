"""
FuXi-S2S Weather Forecast Inference

Main script for running subseasonal-to-seasonal weather forecasts.
Uses utility functions from utils.py for cleaner code organization.
"""

import argparse
import os
import time
import xarray as xr
from glob import glob

from data_util import make_input, print_dataarray
from utils import (
    load_model,
    run_inference,
    land_to_nan,
    print_lite_mode_info,
    print_crop_mode_info,
    print_model_loading_info,
    print_completion_info,
    print_footer
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FuXi-S2S Weather Forecast Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full inference (42 days, 1 member)
  python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --save_dir output

  # Lightweight testing (7 days, 2 members)
  python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --save_dir output --lite

  # Regional output for Philippines
  python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --save_dir output \\
      --crop_lat 13.58 --crop_lon 123.28 --crop_radius 10
        """
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, 
                        help="FuXi-S2S ONNX model file path")
    parser.add_argument('--input', type=str, required=True, 
                        help="Input NetCDF data file path")
    
    # Device settings
    parser.add_argument('--device', type=str, default="cuda", 
                        choices=['cuda', 'cpu'],
                        help="Device to run model on (default: cuda)")
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default="",
                        help="Output directory for forecast files")
    
    # Forecast configuration
    parser.add_argument('--total_step', type=int, default=42, 
                        help="Number of forecast days (default: 42)")
    parser.add_argument('--total_member', type=int, default=1, 
                        help="Number of ensemble members (default: 1)")
    parser.add_argument('--lite', action='store_true', 
                        help="Lightweight testing mode: 7 days, 2 members")
    
    # Regional output cropping
    parser.add_argument('--crop_lat', type=float, default=None, 
                        help="Center latitude for regional output (e.g., 13.58)")
    parser.add_argument('--crop_lon', type=float, default=None, 
                        help="Center longitude for regional output (e.g., 123.28)")
    parser.add_argument('--crop_radius', type=float, default=10.0, 
                        help="Radius in degrees around center (default: 10°)")
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_input_data(input_path):
    """
    Load input data from a file or directory.
    
    Args:
        input_path: Path to a .nc file or directory containing .nc files
        
    Returns:
        xr.DataArray: Combined input data in proper format
    """
    print(f"📂 Loading input data from: {input_path}")
    
    # Check if input_path is a directory
    if os.path.isdir(input_path):
        # Directory with multiple variable files - use make_input to combine
        print(f"  Combining variable files using make_input()...")
        input_data = make_input(input_path)
    else:
        # Single file - open directly
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        input_data = xr.open_dataarray(input_path)
    
    print(f"  ✓ Loaded data shape: {input_data.shape}")
    return input_data


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for FuXi-S2S inference."""
    args = parse_args()
    
    # Apply lightweight mode overrides
    if args.lite:
        args.total_step = 7
        args.total_member = 2
        print_lite_mode_info(args.total_step, args.total_member)
    
    # Display region crop info
    if args.crop_lat is not None and args.crop_lon is not None:
        print_crop_mode_info(args.crop_lat, args.crop_lon, args.crop_radius)
    
    # Load input data
    input_data = load_input_data(args.input)
    print_dataarray(input_data)
    
    # Load model
    print_model_loading_info(args.device)
    start = time.perf_counter()
    model = load_model(args.model, args.device)
    input_names = [inp.name for inp in model.get_inputs()]
    load_time = time.perf_counter() - start
    print(f'✓ Model loaded successfully in {load_time:.1f} seconds')
    print_footer()
    
    # Run inference
    run_inference(
        model=model,
        input_data=input_data,
        total_step=args.total_step,
        total_member=args.total_member,
        input_names=input_names,
        save_dir=args.save_dir,
        crop_lat=args.crop_lat,
        crop_lon=args.crop_lon,
        crop_radius=args.crop_radius,
        print_fn=print_dataarray
    )
    
    # Display completion
    print_completion_info(args.save_dir, args.total_member, args.total_step)


if __name__ == "__main__":
    main()
