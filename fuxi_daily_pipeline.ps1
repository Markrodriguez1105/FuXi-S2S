# Daily FuXi-S2S forecast pipeline for production

param(
    [string]$InitDate = "",  # YYYYMMDD (optional). If empty, auto-select newest available date.
    [int]$Members = 11,
    [string]$Station = "Pacol, Naga City"
)

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Machine Learning\FuXi-S2S\venv_fuxi\Scripts\python.exe"
$PROJECT = "C:\Machine Learning\FuXi-S2S"

$InitDateDisplay = $(if ($InitDate -eq "") { "auto" } else { $InitDate })

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FuXi-S2S Daily Forecast Pipeline" -ForegroundColor Cyan
Write-Host "Init Date: $InitDateDisplay" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Download ERA5 data
Write-Host "`n[1/4] Downloading ERA5 data..." -ForegroundColor Yellow
if ($InitDate -eq "") {
    & $PYTHON "$PROJECT\download_era5.py" --output "data/realtime"
} else {
    & $PYTHON "$PROJECT\download_era5.py" --date $InitDate --output "data/realtime"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERA5 download failed! Exiting." -ForegroundColor Red
    exit 1
}

# Use the actual init date used by the downloader so all steps stay in sync
$InitDateFile = Join-Path "data/realtime" "init_date_used.txt"
if (-not (Test-Path $InitDateFile)) {
    Write-Host "ERROR: Missing data/realtime/init_date_used.txt (downloader did not write it)." -ForegroundColor Red
    exit 1
}
$InitDateUsed = (Get-Content $InitDateFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($InitDateUsed)) {
    Write-Host "ERROR: init_date_used.txt is empty." -ForegroundColor Red
    exit 1
}

Write-Host "Using init date: $InitDateUsed" -ForegroundColor Cyan
$InitDate = $InitDateUsed

if ([string]::IsNullOrWhiteSpace($InitDate)) {
    Write-Host "ERROR: InitDate resolved to empty string; cannot store to MongoDB." -ForegroundColor Red
    exit 1
}

# Step 2: Run inference
Write-Host "`n[2/4] Running FuXi-S2S inference ($($Members) members)..." -ForegroundColor Yellow
& $PYTHON "$PROJECT\inference.py" `
    --model "model/fuxi_s2s.onnx" `
    --input "data/realtime" `
    --save_dir "output" `
    --total_step 42 `
    --total_member $Members `
    --crop_lat 13.58 --crop_lon 123.28 --crop_radius 10
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Inference failed! Exiting." -ForegroundColor Red
    exit 1
}

# Step 3: Store to MongoDB
Write-Host "`n[3/4] Storing forecasts to MongoDB..." -ForegroundColor Yellow
Write-Host "Using init date for MongoDB: $InitDate" -ForegroundColor Cyan
& $PYTHON "$PROJECT\store_forecasts_to_mongo.py" `
    --fuxi_output "output" `
    --init_date "$InitDate" `
    --station $Station `
    --members $Members
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: MongoDB storage failed! Exiting." -ForegroundColor Red
    exit 1
}

# Step 4: Verify
Write-Host "`n[4/4] Verifying MongoDB storage..." -ForegroundColor Yellow
& $PYTHON "$PROJECT\verify_mongo.py"

Write-Host "`nOK: Daily forecast complete!" -ForegroundColor Green