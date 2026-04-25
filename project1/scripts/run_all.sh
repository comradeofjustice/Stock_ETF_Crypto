#!/usr/bin/env bash
# run_all.sh
# ============
# Sequential execution of all pipeline scripts.
# Usage: bash /root/autodl-tmp/CSY/project1/scripts/run_all.sh

set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/CSY/project1"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/output"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/run_log.txt"
echo "=== Pipeline Run: $(date) ===" > "$RUN_LOG"

run_script() {
    local script_name="$1"
    local script_path="$PROJECT_ROOT/scripts/$script_name"
    echo ""
    echo "============================================"
    echo "Running: $script_name"
    echo "============================================"
    echo "[$(date)] Starting $script_name" >> "$RUN_LOG"

    local start_time
    start_time=$(date +%s)

    if python "$script_path" 2>&1 | tee -a "$RUN_LOG"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[$(date)] $script_name completed in ${duration}s" >> "$RUN_LOG"
        echo "✓ $script_name completed in ${duration}s"
    else
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[$(date)] $script_name FAILED after ${duration}s" >> "$RUN_LOG"
        echo "✗ $script_name FAILED after ${duration}s"
        return 1
    fi
}

# Run all scripts in order
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   TTP Stock/ETF/Crypto Analysis Pipeline    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

run_script "01_classify_universe.py"
run_script "02_download_stocks.py"
run_script "03_download_etfs.py"
run_script "04_download_crypto.py"
run_script "05_to_qlib_format.py"
run_script "06_compute_features.py"
run_script "07_generate_reports.py"
run_script "08_generate_index_excel.py"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - config/universe.yaml"
echo "  - data/qlib_data/ (us_stock, us_etf, crypto)"
echo "  - data/features/ (*.parquet)"
echo "  - reports/ (*.html)"
echo "  - output/TTP_Stock_ETF_Index.xlsx"
echo "  - output/download_errors.log"
echo ""
echo "Run log: $RUN_LOG"
