#!/usr/bin/env bash
# Launch the Streamlit training dashboard on 0.0.0.0:8501
# Accessible over the network (e.g. via Tailscale)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec streamlit run "$SCRIPT_DIR/dashboard.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true
