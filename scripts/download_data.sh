#!/usr/bin/env bash
set -euo pipefail

SLUG="${1:-cafa-6-protein-function-prediction}"
DATA_DIR="data/raw"

echo "Downloading competition data: $SLUG"
mkdir -p "$DATA_DIR"

# Check for kaggle CLI
if ! command -v kaggle &>/dev/null; then
    echo "Error: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Accept competition rules first (may fail if already accepted)
kaggle competitions list -s "$SLUG" >/dev/null 2>&1 || true

# Download
kaggle competitions download -c "$SLUG" -p "$DATA_DIR"

# Unzip
echo "Extracting files..."
cd "$DATA_DIR"
for f in *.zip; do
    [ -f "$f" ] && unzip -o "$f" && rm "$f"
done

echo ""
echo "Data downloaded to $DATA_DIR/"
ls -lh
