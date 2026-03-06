# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## What This Does

Two training modes for robot policy learning:

- **PushT (fast)**: Diffusion Policy on a 2D pushing task — state-only observations, 4.4M params, trains in minutes on CPU
- **Transfer Cube (full)**: ACT policy on ALOHA bimanual cube transfer — vision+state, 51M params, trains on MPS/GPU
- **Dashboard**: Live Streamlit monitoring for both tasks

## Quick Start

### Prerequisites

- macOS (Apple Silicon) or Linux
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone this repo
git clone git@github.com:AviZurlo/robot-sim.git
cd robot-sim

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install LeRobot with simulation support
uv pip install 'lerobot[aloha,pusht]' streamlit
```

### Fast Training (PushT)

Train a Diffusion Policy on the PushT 2D pushing task. State-only observations, no video decoding — trains in minutes on CPU.

```bash
# Train for 1000 steps (~4.5 min on CPU)
python scripts/train.py --task pusht --steps 1000

# Evaluate the trained policy
python scripts/evaluate.py --checkpoint outputs/train/diffusion_pusht/last --task pusht

# Or run the full pipeline in one command
python scripts/run_experiment.py --task pusht --steps 1000 --n-episodes 10
```

### Full Training (Transfer Cube)

Train an ACT policy on the ALOHA bimanual cube transfer task. Vision+state observations, requires MPS or GPU.

```bash
# Train for 5000 steps (~60 min on MPS)
python scripts/train.py --task transfer_cube --steps 5000 --device mps

# Evaluate
python scripts/evaluate.py --checkpoint outputs/train/act_transfer_cube/last

# With pretrained baseline comparison
python scripts/run_experiment.py --task transfer_cube --steps 5000 --device mps --baseline
```

### Run Pretrained Simulation

```bash
# Run 3 episodes with the pretrained ACT policy
python scripts/run_sim.py

# Customize
python scripts/run_sim.py --n-episodes 5 --device mps
```

Videos are saved to `outputs/videos/`.

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `transfer_cube` | Task: `pusht` (fast) or `transfer_cube` (full) |
| `--steps` | task-dependent | Total training steps (pusht: 1000, transfer_cube: 5000) |
| `--batch-size` | task-dependent | Batch size (pusht: 64, transfer_cube: 8) |
| `--lr` | task-dependent | Learning rate (pusht: 1e-4, transfer_cube: 1e-5) |
| `--device` | `mps` | Compute device: `cpu`, `mps`, or `cuda` |
| `--output-dir` | task-dependent | Checkpoint directory |
| `--log-freq` | `50` | Log every N steps |
| `--save-freq` | `1000` | Save checkpoint every N steps |
| `--resume` | - | Resume from last checkpoint |
| `--compile` | - | Use torch.compile() (CUDA only; auto-skips on MPS) |

## Evaluation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Local checkpoint path or HuggingFace model ID |
| `--task` | `transfer_cube` | Task: `pusht` or `transfer_cube` |
| `--n-episodes` | `10` | Number of evaluation episodes |
| `--device` | `cpu` | Compute device |
| `--output-dir` | `outputs/eval/<name>` | Output directory |

## Live Dashboard

Real-time Streamlit dashboard for monitoring training experiments. Supports both PushT and Transfer Cube tasks.

```bash
bash scripts/start_dashboard.sh
```

Access at `http://localhost:8501` or `http://<tailscale-ip>:8501`.

| Page | What it shows |
|------|--------------|
| **Live Training** | Auto-refreshing loss curves, task selector |
| **Evaluation Results** | Success rate, rewards, eval videos |
| **Experiment History** | Training checkpoints and evaluation runs for all tasks |
| **Baseline Comparison** | Side-by-side metrics comparison |
| **Status** | Per-task training status, checkpoints, disk usage |

## VLA Probing Dashboard

Interactive Streamlit dashboard for viewing and comparing VLA diagnostic probe results across models.

```bash
bash scripts/start_probe_dashboard.sh
```

Access at `http://localhost:8502` or `http://<tailscale-ip>:8502`.

| Page | What it shows |
|------|--------------|
| **Overview** | Summary heatmap (model × probe), radar chart, model metadata |
| **Per-Probe Details** | Detailed metrics for each of the 8 probes, side-by-side comparison |
| **Metrics Explorer** | Scatter plots, distributions, filterable metric table |
| **Attention Maps** | Attention overlay images and IoU metrics |

Reads results from `outputs/probes/probe_results_<model>.json`. Generate results with:

```bash
python -m vla_probing --model xvla --device mps
```

## Performance Benchmarks

### Device Comparison — ACT Transfer Cube (51M params, batch=8)

| Device | Steps/sec | Relative | Notes |
|--------|-----------|----------|-------|
| CPU (M-series) | 0.5 | 1.0x | Baseline |
| **MPS (Apple Silicon GPU)** | **1.1** | **2.2x** | Default device |
| MPS + torch.compile | ❌ | — | Not supported (Metal buffer limits + missing ops) |
| CPU + torch.compile | TBD | — | Available via `--compile` flag |
| CUDA (A100) | ~10-20 | ~20-40x | Estimated, not tested |

**Key findings:**
- MPS provides a free 2.2x speedup on Apple Silicon with zero code changes
- `torch.compile()` on MPS hits two blockers: Metal shader compiler runs out of buffer slots (inductor backend), and `aten::native_dropout` is not implemented (aot_eager backend falls back to CPU, making it 5x slower)
- `torch.compile()` with inductor backend is available for CUDA GPUs via `--compile`

### Device Comparison — Diffusion PushT (4.4M params, batch=64)

| Device | Steps/sec | Notes |
|--------|-----------|-------|
| CPU | ~4.5 | State-only, no vision — CPU-bound is fine |

## Results

### PushT — Diffusion Policy (1000 steps, CPU)

| Metric | Value |
|--------|------:|
| Loss | 1.34 → 0.066 |
| Training Time | 4.5 min |
| Parameters | 4.4M |
| Success Rate | 0% (10 episodes) |
| Avg Reward | 17.75 |

![Training Loss](outputs/plots/training_loss.png)

![Eval Rewards](outputs/plots/eval_rewards.png)

### Transfer Cube — ACT (500 steps, MPS)

| Metric | Ours (500 steps) | Pretrained (100k steps) |
|--------|----------------:|------------------------:|
| Success Rate | 0% | 80% |
| Avg Reward | 0.00 | 179.80 |
| Training Time | 5 min | ~60+ min |

## lerobot-cache

Transparent caching layer that pre-decodes LeRobot MP4 video frames to safetensors files on disk. During training, frames load from cache instead of decoding video on the fly — up to 30x faster random-access reads.

### Quick Start

```python
from lerobot_cache import CachedDataset

# Drop-in replacement for LeRobotDataset — auto-caches on first use
dataset = CachedDataset("lerobot/aloha_sim_transfer_cube_human")
item = dataset[0]  # loads images from cache, not video
```

### Benchmark

Dataset: `lerobot/aloha_sim_transfer_cube_human` (50 episodes, 20K frames, 480x640)

| Backend | Access | ms/frame | FPS | Speedup |
|---------|--------|----------|-----|---------|
| Video decode | random | 2.1 | 483 | 1.0x |
| **Safetensors** | **random** | **0.1** | **14,605** | **30.2x** |
| NumPy mmap | random | 1.1 | 909 | 1.9x |

### CLI

```bash
# Pre-decode a dataset
lerobot-cache prepare lerobot/aloha_sim_transfer_cube_human

# Run benchmark
lerobot-cache benchmark lerobot/aloha_sim_transfer_cube_human --num-frames 500

# Check cache status
lerobot-cache info

# Clear cache
lerobot-cache clear lerobot/aloha_sim_transfer_cube_human
```

### Training with Cache

```bash
python scripts/train.py --task transfer_cube --use-cache --device mps
```

## Architecture

| | PushT (Fast) | Transfer Cube (Full) |
|---|---|---|
| **Policy** | Diffusion Policy (DDPM) | ACT (Transformers) |
| **Parameters** | 4.4M | 51.6M |
| **Observations** | 2D state (agent position) | RGB camera + 14-DOF joints |
| **Actions** | 2D continuous | 14-DOF continuous |
| **Dataset** | `lerobot/pusht` (206 demos) | `lerobot/aloha_sim_transfer_cube_human` (50 demos) |
| **Environment** | PushT-v0 (PyGame) | AlohaTransferCube-v0 (MuJoCo) |
| **Training Speed** | ~4.5 steps/s (CPU) | ~1.1 steps/s (MPS) |

## Project Structure

```
.
├── lerobot_cache/              # Cache layer package
│   ├── __init__.py             # Exports CachedDataset, SafetensorsBackend
│   ├── cached_dataset.py       # Drop-in LeRobotDataset replacement
│   ├── cache_backend.py        # Safetensors read/write backend
│   └── cli.py                  # CLI: prepare, benchmark, info, clear
├── scripts/
│   ├── train.py                # Train policies (--task, --use-cache)
│   ├── evaluate.py             # Evaluate trained checkpoints
│   ├── benchmark_decode.py     # Benchmark video decode vs cached loading
│   ├── predecode_dataset.py    # Pre-decode dataset to safetensors cache
│   ├── run_experiment.py       # Chain: train → evaluate → visualize
│   ├── visualize.py            # Generate plots from training/eval data
│   ├── dashboard.py            # Live Streamlit monitoring dashboard
│   ├── run_sim.py              # Run pretrained policy in sim
│   ├── start_dashboard.sh      # Launch training dashboard on 0.0.0.0:8501
│   └── start_probe_dashboard.sh # Launch probing dashboard on 0.0.0.0:8502
├── tests/
│   └── test_cache.py           # Cache backend and integration tests
├── vla_probing/
│   ├── dashboard.py            # Probing results Streamlit dashboard
│   ├── run_all.py              # Run all 8 diagnostic probes
│   └── probes/                 # Probe implementations (8 probes)
├── .streamlit/
│   └── config.toml             # Streamlit dark theme config
├── outputs/
│   ├── plots/                  # Visualization PNGs (tracked in git)
│   ├── probes/                 # VLA probe results JSON (git-ignored)
│   ├── train/                  # Training checkpoints (git-ignored)
│   ├── eval/                   # Evaluation results (git-ignored)
│   └── videos/                 # Simulation videos (git-ignored)
├── PROJECT.md                  # Project roadmap and log
├── pyproject.toml              # Python project metadata
└── README.md
```

<!-- RESULTS_START -->
    ## VLA Diagnostic Probing

    Comparing 6 vision-language-action (VLA) models across 8 diagnostic probes in MuJoCo simulation.
    Each probe tests a specific aspect of visual and language grounding — from basic reaching to null-action
    compliance and attention localization.

    ### Models

    | Model | Scene | Valid |
    | :--- | :---: | :---: |
    | **X-VLA** | WidowX | ✓ |
| **OpenVLA** | WidowX | ✓ |
| **Pi0†** | WidowX | † OOD |
| **Pi0** | Franka | ✓ |
| **OpenVLA-OFT** | Franka | ✓ |
| **Cosmos Policy** | Franka | ✓ |

    † Pi0 on WidowX is out-of-distribution (Pi0 was trained on Franka). Results are not valid for comparison.

    ### Radar Summary

    ![Radar comparison](results/figures/radar_comparison.png)

    Each axis is normalized across all models (outward = better). See full metrics table below.

    ### Full Results

    | Metric | X-VLA<br>(WidowX) | OpenVLA<br>(WidowX) | Pi0†<br>(WidowX) | Pi0<br>(Franka) | OpenVLA-OFT<br>(Franka) | Cosmos Policy<br>(Franka) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** |  |  |  |  |  |  |
| Direction Alignment ↑ | 0.866 | N/A | -0.015 | 0.354 | 0.630 | 0.370 |
| Distance to Target ↓ | 0.125 | 0.269 | 0.992 | 0.827 | 0.520 | 0.530 |
| Trajectory Spread ↓ | 0.018 | 0.002 | 0.443 | 0.442 | 0.000 | 0.035 |
| **Spatial Symmetry** |  |  |  |  |  |  |
| Swap Sensitivity ↑ | 1.504 | N/A | N/A | 0.003 | N/A | N/A |
| Endpoint→New Block Pos ↓ | 0.209 | 0.267 | 0.605 | 0.821 | 0.576 | 0.328 |
| **Camera Sensitivity** |  |  |  |  |  |  |
| Mirror Sensitivity ↑ | 0.391 | 0.145 | 1.743 | 0.071 | 0.138 | 0.000 |
| Flip Sensitivity ↑ | 0.297 | 0.042 | 1.739 | 0.008 | 0.126 | 0.127 |
| **View Ablation** |  |  |  |  |  |  |
| Primary Ablated ↑ | 0.385 | 0.145 | 1.739 | 0.101 | 0.135 | 0.077 |
| Secondary Ablated ↑ | 0.124 | 0.042 | 1.739 | 0.303 | 0.192 | 0.268 |
| Fully Blind ↑ | 0.486 | 0.074 | 1.699 | 0.281 | 0.201 | 0.409 |
| **Counterfactual** |  |  |  |  |  |  |
| Synonym Sensitivity ↓ | 0.128 | 0.000 | 0.201 | 0.186 | 0.046 | 0.020 |
| Max Synonym Sens. ↓ | 0.185 | 0.000 | 0.272 | 0.220 | 0.061 | 0.041 |
| **Null Action** |  |  |  |  |  |  |
| Null Displacement ↓ | 0.147 | 0.269 | 0.845 | 0.791 | 0.653 | 0.587 |
| Null/Baseline Ratio ↓ | 0.820 | 1.000 | 1.010 | 1.037 | 0.978 | 1.057 |
| **Perturbation** |  |  |  |  |  |  |
| Mean Sensitivity ↑ | 0.979 | 0.000 | 0.006 | 0.001 | 0.062 | 0.013 |
| Displacement Corr. ↑ | 0.237 | N/A | 0.776 | -0.769 | -0.448 | 0.492 |
| **Attention** |  |  |  |  |  |  |
| Mean IoU ↑ | 0.053 | 0.110 | 0.000 | 0.042 | 0.058 | 0.000 |

    ↑ higher is better · ↓ lower is better · N/A = metric not available for this model/scene

    ### Probe Charts

    | | |
    |:---:|:---:|
    | ![Baseline](results/figures/probe_baseline.png) | ![Spatial Symmetry](results/figures/probe_spatial_symmetry.png) |
    | ![Camera Sensitivity](results/figures/probe_camera_sensitivity.png) | ![View Ablation](results/figures/probe_view_ablation.png) |
    | ![Counterfactual](results/figures/probe_counterfactual.png) | ![Null Action](results/figures/probe_null_action.png) |
    | ![Perturbation](results/figures/probe_perturbation.png) | ![Attention](results/figures/probe_attention.png) |

    ### Key Findings

    - **X-VLA** is the only model with genuine spatial grounding: it re-routes when blocks are swapped
      (swap\_sensitivity 1.50) and responds to block movement (perturbation sensitivity 0.98). However its
      perturbation response is non-linear — it does not scale proportionally with displacement (correlation 0.24).
    - **OpenVLA** runs a near-fixed motor program: zero perturbation sensitivity, identical outputs for all
      synonym phrasings, and null/baseline ratio of exactly 1.0.
    - **Pi0 (Franka)** is highly stochastic (trajectory spread 0.44), making it difficult to isolate genuine
      scene conditioning from noise. Its perturbation correlation is strongly negative (-0.77), suggesting
      out-of-distribution collapse for larger block displacements.
    - **No model passes the null action test.** All models produce significant motion when instructed to
      stay still, with null/baseline ratios ranging from 0.82 (X-VLA, best) to 1.06 (Cosmos Policy, worst).
    - **Attention IoU is near zero for all models**, suggesting that spatial attention in these VLMs does
      not localize to task-relevant objects in a pixel-precise way.

    ### Reproducing Results

    ```bash
    # Run all probes for a model
    python -m vla_probing.run_all --model xvla --scene widowx --device mps

    # Regenerate this section and figures
    python scripts/generate_results.py
    ```

<!-- RESULTS_END -->
