"""
Compare FuXi-S2S Forecast with PAGASA Observations

Main script for comparing weather forecasts with historical observations.
Outputs are saved to: compare/result/[location]/yyyymm.csv
"""

import argparse
from utils import STATIONS, run_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare FuXi-S2S forecasts with PAGASA observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_pagasa.py --pagasa data/pagasa/CBSUA_Pili.xlsx --fuxi_output output

  # With specific initialization date
  python compare_pagasa.py --pagasa data/pagasa/CBSUA_Pili.xlsx --fuxi_output output \\
      --init_date 20230101

  # Custom station location
  python compare_pagasa.py --pagasa data/pagasa/custom.xlsx --fuxi_output output \\
      --station "My Station" --lat 14.5 --lon 121.0

Output:
  Results are saved to: compare/result/[station]/yyyymm.csv
  Example: compare/result/CBSUA_Pili/202301.csv
        """
    )
    
    parser.add_argument('--pagasa', type=str, 
                        default='data/pagasa/CBSUA Pili, Camarines Sur Daily data.xlsx',
                        help='Path to PAGASA Excel file')
    parser.add_argument('--fuxi_output', type=str, default='output',
                        help='Path to FuXi-S2S output directory')
    parser.add_argument('--member', type=int, default=0,
                        help='Ensemble member to use (0-10)')
    parser.add_argument('--init_date', type=str, default=None,
                        help='Initialization date (YYYYMMDD or YYYY/MM/DD or YYYY-MM-DD)')
    parser.add_argument('--station', type=str, default='CBSUA Pili',
                        help='Station name for coordinates and output folder')
    parser.add_argument('--lat', type=float, default=None,
                        help='Override station latitude')
    parser.add_argument('--lon', type=float, default=None,
                        help='Override station longitude')
    parser.add_argument('--output_dir', type=str, default='compare/result',
                        help='Base output directory (default: compare/result)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_comparison(
        pagasa_path=args.pagasa,
        fuxi_output_dir=args.fuxi_output,
        station=args.station,
        member=args.member,
        init_date=args.init_date,
        lat=args.lat,
        lon=args.lon,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
