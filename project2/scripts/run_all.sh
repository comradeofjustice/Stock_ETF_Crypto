#!/bin/bash
# Run the full pipeline from project2/
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Build labels ==="
python scripts/01_build_labels.py

echo "=== Step 2: Build dataset ==="
python scripts/02_build_dataset.py

echo "=== Step 3: Train HAR-RV baseline ==="
python scripts/03_train_har_rv.py

echo "=== Step 4: Train LightGBM ==="
python scripts/04_train_lgbm.py

echo "=== Step 5: Train LSTM ==="
python scripts/05_train_lstm.py

echo "=== Step 6: Train Transformer ==="
python scripts/06_train_transformer.py

echo "=== Step 7: Evaluate & compare ==="
python scripts/07_evaluate.py

echo ""
echo "All done. Results in results/metrics/ and results/plots/"
