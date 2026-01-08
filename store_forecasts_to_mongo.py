"""Store FuXi-S2S station-level forecasts into MongoDB.

Writes two datasets:
- Member-level station forecasts (one doc per member per valid_time)
- Final station forecasts (ensemble mean, optional bias correction)

MongoDB connection:
- Set env var MONGO_DB_URI (or put it in .env), or pass --mongo_uri

Example:
  python store_forecasts_to_mongo.py --fuxi_output output --init_date 20200602 --station "CBSUA Pili" --members 11
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os

import pandas as pd

from utils.compare import STATIONS, load_fuxi_output, extract_station_forecast
from utils.bias_correction import BiasCorrector, ensemble_average
from utils.mongo_store import MongoConfig, MongoForecastStore


VARIABLES_DEFAULT = ["tp", "t2m", "d2m", "msl", "10u", "10v"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _read_env_file_value(dotenv_path: str, key: str) -> str | None:
    """Minimal .env reader (no external dependency).

    Supports lines like:
      KEY=value
      KEY="value"
    Ignores comments and blank lines.
    """
    if not os.path.exists(dotenv_path):
        return None
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != key:
                    continue
                value = v.strip().strip('"').strip("'")
                return value or None
    except Exception:
        return None
    return None


def _get_mongo_uri(cli_value: str | None) -> str:
    if cli_value:
        return cli_value

    # Prefer the repo's .env key first
    env_uri = os.environ.get("MONGO_DB_URI")
    if env_uri:
        return env_uri

    # Try reading from .env in the current working directory
    file_uri = _read_env_file_value(".env", "MONGO_DB_URI")
    if file_uri:
        return file_uri

    raise ValueError("Missing MongoDB URI. Set MONGO_DB_URI or pass --mongo_uri")


def _normalize_init_date(init_date: str) -> str:
    # Accept YYYYMMDD, YYYY/MM/DD, YYYY-MM-DD
    cleaned = init_date.replace("/", "").replace("-", "")
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise ValueError(f"Invalid init_date '{init_date}'. Expected YYYYMMDD.")
    return cleaned


def _build_run_id(init_date: str, station_name: str, num_members: int, use_ensemble: bool, apply_bias_correction: bool, version: str | None) -> str:
    """Build a unique run_id with automatic timestamp for versioning."""
    # Sanitize station name
    station_safe = "".join(ch if ch.isalnum() else "_" for ch in station_name)
    
    # Use provided version or generate timestamp
    if version:
        ver = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in version)
    else:
        ver = _utc_now().strftime("%Y%m%dT%H%M%S")
    
    return f"{init_date}_{station_safe}_m{num_members}_e{int(use_ensemble)}_b{int(apply_bias_correction)}_{ver}"


def _df_row_to_member_doc(row: pd.Series, run_id: str, station_name: str, station_lat: float, station_lon: float, member: int) -> dict:
    valid_time = pd.to_datetime(row["valid_time"])
    init_time = pd.to_datetime(row["init_time"])

    # Use Celsius where possible
    t2m_c = row.get("t2m_celsius")
    if t2m_c is None or pd.isna(t2m_c):
        t2m_k = _safe_float(row.get("t2m"))
        t2m_c = None if t2m_k is None else (t2m_k - 273.15)

    d2m_c = row.get("d2m_celsius")
    if d2m_c is None or pd.isna(d2m_c):
        d2m_k = _safe_float(row.get("d2m"))
        d2m_c = None if d2m_k is None else (d2m_k - 273.15)

    doc = {
        "run_id": run_id,
        "member": int(member),
        "station": {"name": station_name, "lat": float(station_lat), "lon": float(station_lon)},
        "init_time": init_time.to_pydatetime(),
        "lead_time_days": int(_safe_int(row.get("lead_time_days")) or 0),
        "valid_time": valid_time.to_pydatetime(),
        "variables": {
            "tp_mm": _safe_float(row.get("tp")),
            "t2m_c": _safe_float(t2m_c),
            "d2m_c": _safe_float(d2m_c),
            "msl_pa": _safe_float(row.get("msl")),
            "u10": _safe_float(row.get("10u")),
            "v10": _safe_float(row.get("10v")),
            "wind_speed": _safe_float(row.get("wind_speed")),
            "wind_direction_deg": _safe_float(row.get("wind_direction")),
        },
        "quality": {
            "kind": "member",
            "ensemble_mean": False,
            "bias_corrected": False,
        },
    }

    return doc


def _df_row_to_final_doc(row: pd.Series, run_id: str, station_name: str, station_lat: float, station_lon: float, use_ensemble: bool, bias_corrected: bool) -> dict:
    valid_time = pd.to_datetime(row["valid_time"])
    init_time = pd.to_datetime(row["init_time"])

    # Choose corrected columns if present
    tp = row.get("tp_corrected") if "tp_corrected" in row else row.get("tp")

    t2m_c = row.get("t2m_corrected") if "t2m_corrected" in row else row.get("t2m_celsius")
    if t2m_c is None or pd.isna(t2m_c):
        t2m_k = _safe_float(row.get("t2m"))
        t2m_c = None if t2m_k is None else (t2m_k - 273.15)

    d2m_c = row.get("d2m_celsius")
    if d2m_c is None or pd.isna(d2m_c):
        d2m_k = _safe_float(row.get("d2m"))
        d2m_c = None if d2m_k is None else (d2m_k - 273.15)

    wind_speed = row.get("wind_speed_corrected") if "wind_speed_corrected" in row else row.get("wind_speed")

    uncertainty: dict[str, float | None] = {}
    if "tp_std" in row:
        uncertainty["tp_mm_std"] = _safe_float(row.get("tp_std"))
    if "t2m_celsius_std" in row:
        uncertainty["t2m_c_std"] = _safe_float(row.get("t2m_celsius_std"))
    if "wind_speed_std" in row:
        uncertainty["wind_speed_std"] = _safe_float(row.get("wind_speed_std"))

    doc = {
        "run_id": run_id,
        "station": {"name": station_name, "lat": float(station_lat), "lon": float(station_lon)},
        "init_time": init_time.to_pydatetime(),
        "lead_time_days": int(_safe_int(row.get("lead_time_days")) or 0),
        "valid_time": valid_time.to_pydatetime(),
        "variables": {
            "tp_mm": _safe_float(tp),
            "t2m_c": _safe_float(t2m_c),
            "d2m_c": _safe_float(d2m_c),
            "msl_pa": _safe_float(row.get("msl")),
            "u10": _safe_float(row.get("10u")),
            "v10": _safe_float(row.get("10v")),
            "wind_speed": _safe_float(wind_speed),
            "wind_direction_deg": _safe_float(row.get("wind_direction")),
        },
        "uncertainty": uncertainty,
        "quality": {
            "kind": "final",
            "ensemble_mean": bool(use_ensemble),
            "bias_corrected": bool(bias_corrected),
        },
    }

    return doc


def build_station_forecasts(
    fuxi_output: str,
    init_date: str,
    station_lat: float,
    station_lon: float,
    num_members: int,
    variables: list[str],
) -> list[pd.DataFrame]:
    member_dfs: list[pd.DataFrame] = []

    for member in range(num_members):
        fuxi_data = load_fuxi_output(fuxi_output, member=member, init_date=init_date)
        if fuxi_data is None:
            continue

        df = extract_station_forecast(
            fuxi_data,
            station_lat,
            station_lon,
            variables=variables,
        )
        df["member"] = member

        # Precompute Celsius columns for convenience
        if "t2m" in df.columns and "t2m_celsius" not in df.columns:
            df["t2m_celsius"] = df["t2m"] - 273.15
        if "d2m" in df.columns and "d2m_celsius" not in df.columns:
            df["d2m_celsius"] = df["d2m"] - 273.15

        member_dfs.append(df)

    return member_dfs


def main() -> int:
    parser = argparse.ArgumentParser(description="Store FuXi-S2S station forecasts to MongoDB")

    parser.add_argument("--mongo_uri", type=str, default=None, help="MongoDB URI (overrides FUXI_MONGODB_URI)")
    parser.add_argument("--db", type=str, default="arice", help="MongoDB database name")

    parser.add_argument("--fuxi_output", type=str, default="output", help="FuXi output directory")
    parser.add_argument("--init_date", type=str, required=True, help="Initialization date (YYYYMMDD)")

    parser.add_argument("--station", type=str, default="CBSUA Pili", help="Station name")
    parser.add_argument("--lat", type=float, default=None, help="Station latitude (defaults from utils.compare.STATIONS)")
    parser.add_argument("--lon", type=float, default=None, help="Station longitude (defaults from utils.compare.STATIONS)")

    parser.add_argument("--members", type=int, default=11, help="Number of ensemble members")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable ensemble averaging")
    parser.add_argument("--no_correction", action="store_true", help="Disable bias correction")
    parser.add_argument("--bias_params", type=str, default="bias_correction_params.pkl", help="Bias correction params file")

    parser.add_argument("--version", type=str, default=None, help="Version string (default: auto-generated timestamp)")

    args = parser.parse_args()

    init_date = _normalize_init_date(args.init_date)

    # Resolve station coordinates
    station_lat = args.lat
    station_lon = args.lon
    if station_lat is None or station_lon is None:
        if args.station not in STATIONS:
            raise ValueError(
                f"Unknown station '{args.station}'. Provide --lat and --lon, or add it to utils.compare.STATIONS."
            )
        station_lat = STATIONS[args.station]["lat"]
        station_lon = STATIONS[args.station]["lon"]

    use_ensemble = not args.no_ensemble
    apply_bias_correction = not args.no_correction

    run_id = _build_run_id(
        init_date=init_date,
        station_name=args.station,
        num_members=int(args.members),
        use_ensemble=use_ensemble,
        apply_bias_correction=apply_bias_correction,
        version=args.version,
    )

    mongo_uri = _get_mongo_uri(args.mongo_uri)

    store = MongoForecastStore(MongoConfig(uri=mongo_uri, db_name=args.db))
    try:
        store.ensure_indexes()

        # Build member-level station forecasts
        member_dfs = build_station_forecasts(
            fuxi_output=args.fuxi_output,
            init_date=init_date,
            station_lat=float(station_lat),
            station_lon=float(station_lon),
            num_members=int(args.members),
            variables=VARIABLES_DEFAULT,
        )

        if not member_dfs:
            raise RuntimeError(
                f"No member forecasts loaded for init_date={init_date}. Check that '{args.fuxi_output}' contains output/YYYY/YYYYMMDD/member/*.nc"
            )

        # Compute final forecast
        if use_ensemble and len(member_dfs) > 1:
            final_df = ensemble_average(member_dfs)
        else:
            final_df = member_dfs[0].copy()

        bias_corrected = False
        if apply_bias_correction:
            if not os.path.exists(args.bias_params):
                raise FileNotFoundError(
                    f"Bias correction file not found: {args.bias_params}. Run the bias correction training to generate it, or use --no_correction."
                )
            corrector = BiasCorrector()
            corrector.load(args.bias_params)
            final_df = corrector.transform(final_df)
            bias_corrected = True

        # Upsert run metadata
        init_time = pd.to_datetime(final_df["init_time"].iloc[0])
        run_doc = {
            "_id": run_id,
            "init_date": init_date,
            "init_time": init_time.to_pydatetime(),
            "created_at": _utc_now(),
            "source": {
                "output_dir": args.fuxi_output,
                "num_members": int(args.members),
            },
            "postprocess": {
                "ensemble": bool(use_ensemble),
                "bias_correction": bool(bias_corrected),
                "bias_params_file": None if not bias_corrected else args.bias_params,
            },
            "station": {"name": args.station, "lat": float(station_lat), "lon": float(station_lon)},
        }
        store.upsert_run(run_doc)

        # Member docs
        member_docs = []
        for df in member_dfs:
            member = int(df["member"].iloc[0])
            for _, row in df.iterrows():
                member_docs.append(
                    _df_row_to_member_doc(
                        row=row,
                        run_id=run_id,
                        station_name=args.station,
                        station_lat=float(station_lat),
                        station_lon=float(station_lon),
                        member=member,
                    )
                )

        # Final docs
        final_docs = []
        for _, row in final_df.iterrows():
            final_docs.append(
                _df_row_to_final_doc(
                    row=row,
                    run_id=run_id,
                    station_name=args.station,
                    station_lat=float(station_lat),
                    station_lon=float(station_lon),
                    use_ensemble=use_ensemble,
                    bias_corrected=bias_corrected,
                )
            )

        member_written = store.bulk_upsert_member_forecasts(member_docs)
        final_written = store.bulk_upsert_final_forecasts(final_docs)

        print("=" * 80)
        print("✅ MongoDB write complete")
        print("=" * 80)
        print(f"DB: {args.db}")
        print(f"Run ID: {run_id}")
        print(f"Station: {args.station} ({station_lat}, {station_lon})")
        print(f"Member docs upserted/modified: {member_written}")
        print(f"Final docs upserted/modified:  {final_written}")

        return 0

    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
