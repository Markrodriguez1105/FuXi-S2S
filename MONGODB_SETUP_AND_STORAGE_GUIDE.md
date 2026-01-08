# MongoDB Storage for FuXi-S2S Forecasts

Store station-level weather forecasts in MongoDB for serving to other applications.

---

## Quick Start (Windows + Docker)

### 1. Start MongoDB

```powershell
docker run --name fuxi-mongo -d -p 27017:27017 `
  -e MONGO_INITDB_ROOT_USERNAME=root `
  -e MONGO_INITDB_ROOT_PASSWORD=admin `
  -v fuxi_mongo_data:/data/db `
  mongo:7
```

### 2. Configure connection

Create `.env` in project root:

```
MONGO_DB_URI=...
```

### 3. Run ingestion

```powershell
python store_forecasts_to_mongo.py --init_date 20200602 --station "Naga City" --members 11 --db arice
```

### 4. Verify

```powershell
mongosh [MONGO_DB_URI] --eval "use arice; db.station_forecasts_final.countDocuments()"
```

---

## What Gets Stored

The ingestion script reads NetCDF outputs and writes to MongoDB:

| Collection                 | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| `forecast_runs`            | Metadata per run (init date, config, timestamp)        |
| `station_forecasts_member` | Per-member forecasts (for diagnostics)                 |
| `station_forecasts_final`  | Ensemble mean + bias-corrected forecasts (for serving) |

---

## Data Flow

```
output/YYYY/YYYYMMDD/member/XX/*.nc
        ↓
   load_fuxi_output() → extract_station_forecast()
        ↓
   ensemble_average() (if --members > 1)
        ↓
   BiasCorrector.transform() (if bias_correction_params.pkl exists)
        ↓
   MongoDB: station_forecasts_final
```

---

## Run ID Versioning

Each ingestion creates a unique `run_id`:

```
20200602_CBSUA_Pili_m11_e1_b1_20260108T143022
         │         │   │  │  └── timestamp (auto)
         │         │   │  └── bias correction (1=on)
         │         │   └── ensemble (1=on)
         │         └── members
         └── init date
```

- Re-running creates a **new version** (old data kept)
- Use `--version v1` to set a custom version string

---

## CLI Options

```
python store_forecasts_to_mongo.py [OPTIONS]

Required:
  --init_date YYYYMMDD    Forecast initialization date

Optional:
  --station NAME          Station name (default: CBSUA Pili)
  --lat FLOAT             Custom latitude
  --lon FLOAT             Custom longitude
  --members N             Ensemble members (default: 11)
  --db NAME               Database name (default: fuxi_s2s)
  --no_ensemble           Skip ensemble averaging
  --no_correction         Skip bias correction
  --bias_params FILE      Custom bias params file
  --version STRING        Custom version (default: auto timestamp)
  --mongo_uri URI         Override connection string
```

---

## Document Schema

### `station_forecasts_final`

```json
{
  "run_id": "20200602_CBSUA_Pili_m11_e1_b1_20260108T143022",
  "station": { "name": "CBSUA Pili", "lat": 13.58, "lon": 123.28 },
  "init_time": "2020-06-02T00:00:00Z",
  "lead_time_days": 10,
  "valid_time": "2020-06-12T00:00:00Z",
  "variables": {
    "tp_mm": 12.3,
    "t2m_c": 29.8,
    "d2m_c": 24.1,
    "msl_pa": 101200.0,
    "wind_speed": 1.24,
    "wind_direction_deg": 250.0
  },
  "quality": { "ensemble_mean": true, "bias_corrected": true }
}
```

---

## Indexes (auto-created)

```javascript
// Final forecasts
db.station_forecasts_final.createIndex(
  { run_id: 1, "station.name": 1, valid_time: 1 },
  { unique: true }
);
db.station_forecasts_final.createIndex({ "station.name": 1, valid_time: 1 });

// Member forecasts
db.station_forecasts_member.createIndex(
  { run_id: 1, "station.name": 1, member: 1, valid_time: 1 },
  { unique: true }
);
```

---

## Common Queries

### Latest forecast for a station

```javascript
db.station_forecasts_final
  .find({ "station.name": "CBSUA Pili" })
  .sort({ valid_time: 1 })
  .limit(42);
```

### Get all run versions

```javascript
db.forecast_runs.find().sort({ created_at: -1 });
```

### Delete old runs

```javascript
db.forecast_runs.deleteOne({ _id: "old_run_id" });
db.station_forecasts_final.deleteMany({ run_id: "old_run_id" });
db.station_forecasts_member.deleteMany({ run_id: "old_run_id" });
```

---

## Production Tips

1. **Use read-only user for API**: Create a separate user with `read` role for your forecast API
2. **Archive old runs**: Delete or archive runs older than N days
3. **Monitor disk usage**: Member forecasts can be large; consider storing only `final` in production
4. **Use connection pooling**: Set `maxPoolSize` in your API's MongoDB client
