#!/bin/bash
# ==============================================================================
# FuXi-S2S Docker Entrypoint Script
# ==============================================================================
# This script handles initialization and command routing for the container
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  FuXi-S2S Weather Forecast Container  ${NC}"
echo -e "${CYAN}========================================${NC}"

# ==============================================================================
# Normalize environment variables
# ==============================================================================
normalize_env() {
    # store_forecasts_to_mongo.py expects MONGO_DB_URI
    if [[ -z "${MONGO_DB_URI:-}" && -n "${MONGO_URI:-}" ]]; then
        export MONGO_DB_URI="$MONGO_URI"
    fi
    # Keep compatibility the other way too
    if [[ -z "${MONGO_URI:-}" && -n "${MONGO_DB_URI:-}" ]]; then
        export MONGO_URI="$MONGO_DB_URI"
    fi
}

# ==============================================================================
# Setup CDS API credentials
# ==============================================================================
setup_cds_credentials() {
    # Check if .cdsapirc is already mounted (takes priority)
    if [[ -f /root/.cdsapirc ]]; then
        echo -e "${GREEN}[Setup] Using mounted CDS API credentials file${NC}"
    elif [[ -n "$CDS_API_KEY" && -n "$CDS_API_URL" ]]; then
        # Write credentials from environment variables
        echo -e "${GREEN}[Setup] Configuring CDS API credentials from environment...${NC}"
        cat > /root/.cdsapirc << EOF
url: $CDS_API_URL
key: $CDS_API_KEY
EOF
        echo -e "${GREEN}[Setup] CDS API configured successfully${NC}"
    else
        echo -e "${YELLOW}[Warning] No CDS API credentials found!${NC}"
        echo -e "${YELLOW}          Set CDS_API_URL and CDS_API_KEY environment variables,${NC}"
        echo -e "${YELLOW}          or mount .cdsapirc file to /root/.cdsapirc${NC}"
    fi
}

# ==============================================================================
# Verify GPU availability
# ==============================================================================
check_gpu() {
    echo -e "${CYAN}[GPU Check] Detecting NVIDIA GPU...${NC}"
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null; then
        echo -e "${GREEN}[GPU Check] GPU detection complete${NC}"
    else
        echo -e "${YELLOW}[Warning] GPU not detected, falling back to CPU${NC}"
    fi
}

# ==============================================================================
# Display help
# ==============================================================================
show_help() {
    echo ""
    echo -e "${GREEN}FuXi-S2S Docker Commands:${NC}"
    echo ""
    echo "  ${CYAN}pipeline${NC}              - Run full forecast pipeline (download → inference → store)"
    echo "  ${CYAN}download${NC}              - Download ERA5 data only"
    echo "  ${CYAN}inference${NC}             - Run model inference only"
    echo "  ${CYAN}store${NC}                 - Store forecasts to MongoDB only"
    echo "  ${CYAN}verify${NC}                - Verify MongoDB storage"
    echo "  ${CYAN}shell${NC}                 - Start interactive bash shell"
    echo "  ${CYAN}python${NC}                - Start Python interpreter"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo ""
    echo "  # Run full pipeline with auto date detection"
    echo "  docker-compose run fuxi-s2s pipeline"
    echo ""
    echo "  # Run pipeline for specific date"
    echo "  docker-compose run fuxi-s2s pipeline --date 20250109"
    echo ""
    echo "  # Run with custom parameters"
    echo "  docker-compose run fuxi-s2s pipeline --date 20250109 --members 11 --steps 42"
    echo ""
    echo "  # Run inference only (data already downloaded)"
    echo "  docker-compose run fuxi-s2s inference --members 11"
    echo ""
    echo "  # Download ERA5 data only"
    echo "  docker-compose run fuxi-s2s download --date 20250109"
    echo ""
    echo "  # Interactive shell"
    echo "  docker-compose run fuxi-s2s shell"
    echo ""
}

# ==============================================================================
# Run full pipeline
# ==============================================================================
run_pipeline() {
    echo -e "${CYAN}[Pipeline] Starting FuXi-S2S Daily Forecast Pipeline...${NC}"
    
    # Parse arguments
    INIT_DATE=""
    MEMBERS=11
    STEPS=42
    STATION="Pacol, Naga City"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --date)
                INIT_DATE="$2"
                shift 2
                ;;
            --members)
                MEMBERS="$2"
                shift 2
                ;;
            --steps)
                STEPS="$2"
                shift 2
                ;;
            --station)
                STATION="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Init Date: ${INIT_DATE:-auto}"
    echo "  Members:   $MEMBERS"
    echo "  Steps:     $STEPS"
    echo "  Station:   $STATION"
    
    # Step 1: Download ERA5 data
    echo -e "\n${YELLOW}[1/4] Downloading ERA5 data...${NC}"
    if [[ -n "$INIT_DATE" ]]; then
        python download_era5.py --date "$INIT_DATE" --output data/realtime
    else
        python download_era5.py --output data/realtime
    fi
    
    # Get actual init date used
    if [[ -f "data/realtime/init_date_used.txt" ]]; then
        INIT_DATE=$(cat data/realtime/init_date_used.txt | tr -d '[:space:]')
        echo -e "${GREEN}Using init date: $INIT_DATE${NC}"
    else
        echo -e "${RED}ERROR: init_date_used.txt not found${NC}"
        exit 1
    fi
    
    # Step 2: Run inference
    echo -e "\n${YELLOW}[2/4] Running FuXi-S2S inference ($MEMBERS members, $STEPS steps)...${NC}"
    python inference.py \
        --model model/fuxi_s2s.onnx \
        --input data/realtime \
        --save_dir output \
        --total_step "$STEPS" \
        --total_member "$MEMBERS" \
        --crop_lat 13.58 --crop_lon 123.28 --crop_radius 10
    
    # Step 3: Store to MongoDB
    echo -e "\n${YELLOW}[3/4] Storing forecasts to MongoDB...${NC}"
    python store_forecasts_to_mongo.py \
        --fuxi_output output \
        --init_date "$INIT_DATE" \
        --station "$STATION" \
        --members "$MEMBERS"
    
    # Step 4: Verify
    echo -e "\n${YELLOW}[4/4] Verifying MongoDB storage...${NC}"
    python verify_mongo.py
    
    echo -e "\n${GREEN}✓ Pipeline complete!${NC}"
}

# ==============================================================================
# Run download only
# ==============================================================================
run_download() {
    echo -e "${CYAN}[Download] Downloading ERA5 data...${NC}"
    
    INIT_DATE=""
    OUTPUT="data/realtime"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --date)
                INIT_DATE="$2"
                shift 2
                ;;
            --output)
                OUTPUT="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [[ -n "$INIT_DATE" ]]; then
        python download_era5.py --date "$INIT_DATE" --output "$OUTPUT"
    else
        python download_era5.py --output "$OUTPUT"
    fi
    
    echo -e "${GREEN}✓ Download complete!${NC}"
}

# ==============================================================================
# Run inference only
# ==============================================================================
run_inference() {
    echo -e "${CYAN}[Inference] Running FuXi-S2S model inference...${NC}"
    
    MEMBERS=11
    STEPS=42
    INPUT="data/realtime"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --members)
                MEMBERS="$2"
                shift 2
                ;;
            --steps)
                STEPS="$2"
                shift 2
                ;;
            --input)
                INPUT="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    python inference.py \
        --model model/fuxi_s2s.onnx \
        --input "$INPUT" \
        --save_dir output \
        --total_step "$STEPS" \
        --total_member "$MEMBERS" \
        --crop_lat 13.58 --crop_lon 123.28 --crop_radius 10
    
    echo -e "${GREEN}✓ Inference complete!${NC}"
}

# ==============================================================================
# Run store to MongoDB
# ==============================================================================
run_store() {
    echo -e "${CYAN}[Store] Storing forecasts to MongoDB...${NC}"
    
    # Get init date from file if not specified
    if [[ -f "data/realtime/init_date_used.txt" ]]; then
        INIT_DATE=$(cat data/realtime/init_date_used.txt | tr -d '[:space:]')
    else
        INIT_DATE=""
    fi
    
    MEMBERS=11
    STATION="Pacol, Naga City"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --date)
                INIT_DATE="$2"
                shift 2
                ;;
            --members)
                MEMBERS="$2"
                shift 2
                ;;
            --station)
                STATION="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [[ -z "$INIT_DATE" ]]; then
        echo -e "${RED}ERROR: No init date specified and init_date_used.txt not found${NC}"
        exit 1
    fi
    
    python store_forecasts_to_mongo.py \
        --fuxi_output output \
        --init_date "$INIT_DATE" \
        --station "$STATION" \
        --members "$MEMBERS"
    
    echo -e "${GREEN}✓ Storage complete!${NC}"
}

# ==============================================================================
# Main command router
# ==============================================================================

# Setup CDS credentials
normalize_env
setup_cds_credentials

# Route to appropriate command
case "${1:-}" in
    pipeline)
        shift
        check_gpu
        run_pipeline "$@"
        ;;
    download)
        shift
        run_download "$@"
        ;;
    inference)
        shift
        check_gpu
        run_inference "$@"
        ;;
    store)
        shift
        run_store "$@"
        ;;
    verify)
        python verify_mongo.py
        ;;
    shell)
        exec /bin/bash
        ;;
    python)
        shift
        exec python "$@"
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        # If first argument looks like a Python script, run it
        if [[ "${1:-}" == *.py ]]; then
            exec python "$@"
        else
            show_help
        fi
        ;;
esac
