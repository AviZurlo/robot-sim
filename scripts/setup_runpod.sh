#!/usr/bin/env bash
# Setup script for running Cosmos Policy probes on RunPod (CUDA GPU).
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

echo "=== Cosmos Policy RunPod Setup ==="

# 1. Install our project deps
echo "Installing robot-sim dependencies..."
pip install -e ".[vla-probing]"

# 2. Clone and install Cosmos Policy
COSMOS_DIR="/workspace/cosmos-policy"
if [[ ! -d "$COSMOS_DIR" ]]; then
    echo "Cloning Cosmos Policy..."
    git clone https://github.com/nvlabs/cosmos-policy.git "$COSMOS_DIR"
fi

echo "Installing Cosmos Policy..."
cd "$COSMOS_DIR"
pip install -e .

# 3. Install CUDA-specific deps that Cosmos needs
echo "Installing flash-attn and triton..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install may need manual intervention"
pip install triton

# 4. Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 5. Verify Cosmos Policy is importable
cd /workspace/robot-sim
python -c "
import sys
sys.path.insert(0, '/workspace/cosmos-policy')
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
print('Cosmos Policy imported successfully!')
"

echo ""
echo "=== Setup Complete ==="
echo "Run probes with:"
echo "  cd /workspace/robot-sim"
echo "  python -m vla_probing --model cosmos_policy --scene franka --device cuda"
echo ""
echo "Results will be saved to: outputs/probes/probe_results_cosmos_policy_franka.json"
