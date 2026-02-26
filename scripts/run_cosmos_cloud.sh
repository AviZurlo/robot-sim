#!/bin/bash
# =============================================================================
# Cosmos Policy Probing — Cloud GPU Setup Script
# =============================================================================
#
# Run this on a RunPod GPU instance (A40 or A100).
#
# PREREQUISITES:
#   - RunPod account with credits ($10 minimum)
#   - Deploy a "RunPod PyTorch 2.4" pod with an A40 ($0.79/hr) or A100
#   - Open the Web Terminal
#
# USAGE:
#   git clone https://github.com/AviZurlo/robot-sim.git
#   cd robot-sim
#   bash scripts/run_cosmos_cloud.sh
#
# WHAT THIS DOES:
#   1. Installs Cosmos Policy via Docker (their recommended method)
#   2. Downloads the LIBERO checkpoint (~7GB)
#   3. Builds our Cosmos Policy adapter
#   4. Runs all 8 probes on the Franka scene
#   5. Outputs results JSON you can copy back
#
# ESTIMATED TIME: ~15-20 min (mostly model download)
# ESTIMATED COST: ~$0.50-1.00
# =============================================================================

set -e

echo "============================================"
echo "  Cosmos Policy Probing — Cloud GPU Setup"
echo "============================================"
echo ""

# --- Verify GPU ---
echo "[1/7] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU detected. Make sure you launched a GPU pod."
    exit 1
}
echo ""

# --- Install system deps ---
echo "[2/7] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git-lfs > /dev/null 2>&1
pip install uv > /dev/null 2>&1

# --- Clone Cosmos Policy ---
echo "[3/7] Cloning Cosmos Policy..."
if [ ! -d "/tmp/cosmos-policy" ]; then
    git clone --depth 1 https://github.com/nvlabs/cosmos-policy.git /tmp/cosmos-policy
fi

# --- Install Cosmos Policy deps ---
echo "[4/7] Installing Cosmos Policy (this takes a few minutes)..."
cd /tmp/cosmos-policy

# Create venv with Python 3.10 (cosmos-policy requirement)
uv venv .venv --python 3.10 2>/dev/null || python3.10 -m venv .venv 2>/dev/null || {
    echo "Python 3.10 not found, trying system python..."
    python3 -m venv .venv
}
source .venv/bin/activate

# Install cosmos-policy
uv pip install -e "." 2>&1 | tail -3

# Install our additional deps (MuJoCo for scene rendering)
uv pip install mujoco dtw-python 2>&1 | tail -1

# --- Download model checkpoint ---
echo "[5/7] Downloading Cosmos Policy LIBERO checkpoint..."
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('nvidia/Cosmos-Policy-LIBERO-Predict2-2B')
print(f'Checkpoint at: {path}')
"

# --- Set up our probing code ---
echo "[6/7] Setting up probe runner..."
cd /workspace
if [ ! -d "robot-sim" ]; then
    git clone https://github.com/AviZurlo/robot-sim.git
fi
cd robot-sim

# Link cosmos-policy into our Python path
export PYTHONPATH="/tmp/cosmos-policy:$PYTHONPATH"

# --- Run probes ---
echo "[7/7] Running all 8 probes with Cosmos Policy on Franka scene..."
echo ""

# Use the cosmos-policy venv but run our code
PYTHONPATH="/tmp/cosmos-policy:$(pwd):$PYTHONPATH" \
    /tmp/cosmos-policy/.venv/bin/python -m vla_probing \
    --model cosmos_policy --scene franka --device cuda

echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
echo ""

# Display results
python -c "
import json
with open('outputs/probes/probe_results_cosmos_policy_franka.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2))
" 2>/dev/null || echo "Results file not found — check for errors above"

echo ""
echo "============================================"
echo "  NEXT STEPS"  
echo "============================================"
echo ""
echo "1. Copy the results JSON above"
echo "2. Or download the file:"
echo "   cat outputs/probes/probe_results_cosmos_policy_franka.json"
echo ""
echo "3. IMPORTANT: Shut down your RunPod pod to stop billing!"
echo "   Go to runpod.io → Pods → Stop/Terminate"
echo ""
echo "============================================"
