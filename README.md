# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## What This Does

Runs a pretrained **ACT (Action Chunking with Transformers)** policy in a MuJoCo simulation of the [ALOHA bimanual robot](https://tonyzhaozh.github.io/aloha/), performing a cube transfer task. The policy was trained on human demonstrations and achieves ~83% success rate on the task.

## Quick Start

### Prerequisites

- macOS (Apple Silicon) or Linux
- Python 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone this repo
git clone git@github.com:AviZurlo/robot-sim.git
cd robot-sim

# Create virtual environment with Python 3.10
uv venv .venv --python 3.10
source .venv/bin/activate

# Clone and install LeRobot with MuJoCo/Aloha simulation support
git clone https://github.com/huggingface/lerobot.git
uv pip install -e "lerobot[aloha]"
```

### Run Simulation

```bash
# Run 3 episodes with the pretrained ACT policy (default)
python scripts/run_sim.py

# Customize the run
python scripts/run_sim.py --n-episodes 5 --device mps
python scripts/run_sim.py --policy lerobot/act_aloha_sim_insertion_human --task AlohaInsertion-v0
```

Videos are saved to `outputs/videos/` along with `eval_metrics.json`.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | `lerobot/act_aloha_sim_transfer_cube_human` | HuggingFace model ID |
| `--task` | `AlohaTransferCube-v0` | Gymnasium environment task |
| `--n-episodes` | `3` | Number of episodes to run |
| `--device` | `cpu` | Compute device: `cpu`, `mps`, or `cuda` |
| `--output-dir` | `outputs/videos` | Video output directory |
| `--seed` | `1000` | Starting random seed |

## Architecture

- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) v0.4.4 (Hugging Face)
- **Simulation:** [MuJoCo](https://mujoco.org/) 3.5 via [gym-aloha](https://github.com/huggingface/gym-aloha)
- **Policy:** ACT (Action Chunking with Transformers) - 51M params
- **Environment:** `AlohaTransferCube-v0` - bimanual robot transfers a cube between grippers
- **Observation:** RGB camera (480x640) + 14-DOF joint positions
- **Action space:** 14-DOF continuous (7 per arm)

## Available Pretrained Models

| Model | Task | Environment |
|-------|------|-------------|
| `lerobot/act_aloha_sim_transfer_cube_human` | Cube transfer | `AlohaTransferCube-v0` |
| `lerobot/act_aloha_sim_insertion_human` | Peg insertion | `AlohaInsertion-v0` |

## Project Structure

```
.
├── scripts/
│   └── run_sim.py          # Main simulation runner
├── lerobot/                 # LeRobot source (git-ignored, cloned during setup)
├── outputs/videos/          # Generated videos and metrics (git-ignored)
├── PROJECT.md               # Project roadmap and log
├── pyproject.toml           # Python project metadata
└── README.md
```
