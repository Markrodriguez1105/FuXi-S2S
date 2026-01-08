import cdsapi
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Pressure level variables (13 levels each)
PRESSURE_VARS = ['geopotential', 'temperature', 'u_component_of_wind', 
                 'v_component_of_wind', 'specific_humidity']
PRESSURE_LEVELS = ['50', '100', '150', '200', '250', '300', '400', 
                   '500', '600', '700', '850', '925', '1000']

# Surface variables
SURFACE_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'sea_surface_temperature',
                'top_net_thermal_radiation', '10m_u_component_of_wind', '10m_v_component_of_wind',
                '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_sea_level_pressure',
                'total_column_water_vapour', 'total_precipitation']


def get_cds_client():
    """Create CDS API client using local .cdsapirc file."""
    local_rc = PROJECT_ROOT / ".cdsapirc"
    if local_rc.exists():
        # Read credentials from local file
        config = {}
        with open(local_rc) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
        # CDS has used both legacy "UID:API_KEY" and newer token-style keys.
        # Accept either format and let the server validate it.
        return cdsapi.Client(url=config.get("url"), key=config.get("key"))
    else:
        # Fallback to default (~/.cdsapirc)
        return cdsapi.Client()


def _requested_dates(target_date: datetime) -> list[str]:
    return [
        (target_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        target_date.strftime("%Y-%m-%d"),
    ]


def _probe_has_two_consecutive_days(nc_path: str) -> bool:
    ds = xr.open_dataset(nc_path)
    time_coord = None
    if "valid_time" in ds.coords:
        time_coord = ds["valid_time"].values
    elif "time" in ds.coords:
        time_coord = ds["time"].values
    else:
        return False

    if len(time_coord) < 2:
        return False

    t0 = np.datetime64(time_coord[-2])
    t1 = np.datetime64(time_coord[-1])
    return (t1 - t0) == np.timedelta64(1, "D")


def download_era5_for_date(target_date, output_dir='data/realtime', client=None):
    """
    Download ERA5 data for a specific initialization date.
    Requires 2 consecutive days (yesterday + today) for model input.
    """
    os.makedirs(output_dir, exist_ok=True)
    client = client or get_cds_client()
    
    dates = _requested_dates(target_date)

    # Probe using one variable first so we can fail fast when ERA5 isn't available yet.
    probe_path = os.path.join(output_dir, "geopotential.nc")
    print(f"Probing ERA5 availability for dates: {dates[0]} and {dates[1]} ...")
    client.retrieve('reanalysis-era5-pressure-levels', {
        'product_type': 'reanalysis',
        'variable': 'geopotential',
        'pressure_level': PRESSURE_LEVELS,
        'date': dates,
        'time': '00:00',
        'format': 'netcdf',
    }, probe_path)

    if not _probe_has_two_consecutive_days(probe_path):
        # Leave the file (it can help debugging), but signal to caller that this date isn't usable.
        raise RuntimeError(
            "ERA5 download does not contain 2 consecutive timesteps (hist + init). "
            "This usually means the latest day isn't available yet."
        )
    
    # Download remaining pressure level data (skip geopotential because we already downloaded it)
    for var in [v for v in PRESSURE_VARS if v != 'geopotential']:
        output_file = os.path.join(output_dir, f"{var}.nc")
        print(f"Downloading {var}...")
        
        client.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'variable': var,
            'pressure_level': PRESSURE_LEVELS,
            'date': dates,
            'time': '00:00',
            'format': 'netcdf',
        }, output_file)
    
    # Download surface data
    for var in SURFACE_VARS:
        output_file = os.path.join(output_dir, f"{var}.nc")
        print(f"Downloading {var}...")
        
        client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': var,
            'date': dates,
            'time': '00:00',
            'format': 'netcdf',
        }, output_file)
    
    print(f"✅ Downloaded all ERA5 data to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ERA5 data for FuXi-S2S")
    parser.add_argument("--date", type=str, default="",
                        help="Initialization date YYYYMMDD (optional; if omitted the script auto-selects the newest available date)")
    parser.add_argument("--output", type=str, default="data/realtime",
                        help="Output directory (default: data/realtime)")
    parser.add_argument("--lag_days", type=int, default=5,
                        help="How many days back from today to start searching (default: 5, ERA5 latency)")
    parser.add_argument("--max_lookback_days", type=int, default=14,
                        help="Max extra days to search backwards if data isn't available (default: 14)")
    args = parser.parse_args()

    if args.date:
        start_date = datetime.strptime(args.date, "%Y%m%d")
    else:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - timedelta(days=int(args.lag_days))

    client = get_cds_client()
    last_error = None

    for offset in range(0, int(args.max_lookback_days) + 1):
        candidate = start_date - timedelta(days=offset)
        candidate_str = candidate.strftime("%Y%m%d")
        print(f"\nTrying init_date={candidate_str} (offset={offset}d)...")
        try:
            download_era5_for_date(candidate, args.output, client=client)

            init_date_file = os.path.join(args.output, "init_date_used.txt")
            with open(init_date_file, "w", encoding="utf-8") as f:
                f.write(candidate_str + "\n")

            print(f"\n✅ INIT_DATE_USED={candidate_str}")
            raise SystemExit(0)
        except Exception as e:
            last_error = e
            print(f"  Not usable: {e}")

    raise SystemExit(
        f"Failed to download a usable ERA5 init date within {args.max_lookback_days} days. "
        f"Last error: {last_error}"
    )