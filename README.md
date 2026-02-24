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
| `--device` | `cpu` | Compute device: `cpu`, `mps`, or `cuda` |
| `--output-dir` | task-dependent | Checkpoint directory |
| `--log-freq` | `50` | Log every N steps |
| `--save-freq` | `1000` | Save checkpoint every N steps |
| `--resume` | - | Resume from last checkpoint |

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

## Architecture

| | PushT (Fast) | Transfer Cube (Full) |
|---|---|---|
| **Policy** | Diffusion Policy (DDPM) | ACT (Transformers) |
| **Parameters** | 4.4M | 51.6M |
| **Observations** | 2D state (agent position) | RGB camera + 14-DOF joints |
| **Actions** | 2D continuous | 14-DOF continuous |
| **Dataset** | `lerobot/pusht` (206 demos) | `lerobot/aloha_sim_transfer_cube_human` (50 demos) |
| **Environment** | PushT-v0 (PyGame) | AlohaTransferCube-v0 (MuJoCo) |
| **Training Speed** | ~4.5 steps/s (CPU) | ~1.4 steps/s (MPS) |

## Project Structure

```
.
├── scripts/
│   ├── train.py            # Train policies (--task pusht or transfer_cube)
│   ├── evaluate.py         # Evaluate trained checkpoints
│   ├── run_experiment.py   # Chain: train → evaluate → visualize
│   ├── visualize.py        # Generate plots from training/eval data
│   ├── dashboard.py        # Live Streamlit monitoring dashboard
│   ├── run_sim.py          # Run pretrained policy in sim
│   └── start_dashboard.sh  # Launch dashboard on 0.0.0.0:8501
├── .streamlit/
│   └── config.toml          # Streamlit dark theme config
├── outputs/
│   ├── plots/               # Visualization PNGs (tracked in git)
│   ├── train/               # Training checkpoints and loss logs (git-ignored)
│   │   ├── diffusion_pusht/     # PushT training outputs
│   │   └── act_transfer_cube/   # ALOHA training outputs
│   ├── eval/                # Evaluation videos and metrics (git-ignored)
│   └── videos/              # Pretrained policy videos (git-ignored)
├── PROJECT.md               # Project roadmap and log
├── pyproject.toml           # Python project metadata
└── README.md
```
