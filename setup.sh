#!/usr/bin/env bash
set -euo pipefail

echo "=== CAFA 6 Project Setup ==="

# Create directories
mkdir -p data/raw data/processed data/embeddings models submissions

# Create virtual environment
if command -v uv &>/dev/null; then
    echo "Using uv..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[notebooks,dev]"
else
    echo "Using pip..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[notebooks,dev]"
fi

echo ""
echo "=== Environment created ==="
echo "Activate with: source .venv/bin/activate"
echo ""

# Check Kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "WARNING: Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    echo "To set up:"
    echo "  1. Go to https://www.kaggle.com/settings -> API -> Create New Token"
    echo "  2. Place the downloaded kaggle.json in ~/.kaggle/"
    echo "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
fi

echo "To download competition data, run:"
echo "  bash scripts/download_data.sh"
echo ""
echo "Setup complete!"
