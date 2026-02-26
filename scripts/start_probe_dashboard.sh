#!/usr/bin/env bash
# Launch the VLA probing dashboard on port 8502.
# The training dashboard runs on 8501; this avoids conflicts.
set -euo pipefail

cd "$(dirname "$0")/.."
exec streamlit run vla_probing/dashboard.py \
    --server.address 0.0.0.0 \
    --server.port 8502
