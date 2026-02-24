# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## Status: Fast Training Mode

**Current phase:** Two training modes — fast state-based PushT (minutes on CPU) and full vision-based ALOHA transfer cube. Streamlit dashboard supports both tasks.

## Architecture

- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face)
- **Simulation:** Gymnasium + MuJoCo (ALOHA) + PyGame (PushT)
- **Models:** ACT, Diffusion Policy, TDMPC, VQ-BeT (pretrained available)
- **Hardware:** Mac mini M-series, 32GB RAM (sim only, no physical robot)

## Goals

1. Get a simulated robot environment running
2. Run a pretrained policy — see it perform a task
3. Collect simulation data and train a custom policy
4. Iterate: more data → better performance

## Project Log

| Date | What Changed |
|------|-------------|
| 2026-02-24 | Project created. LeRobot selected as framework. |
| 2026-02-24 | LeRobot v0.4.4 + MuJoCo 3.5 + gym-aloha installed. ACT policy running in sim (100% success on cube transfer). |
| 2026-02-24 | Training pipeline: train.py + evaluate.py. ACT trains from scratch on `lerobot/aloha_sim_transfer_cube_human` (50 demos, 20k frames). Loss drops from ~100 to ~0.20 over 5000 steps on MPS. |
| 2026-02-24 | Results dashboard: visualize.py + run_experiment.py. 500-step training run: loss 101→2.7, 0% success (vs 80% pretrained baseline). Plots in `outputs/plots/`. |
| 2026-02-24 | Live Streamlit dashboard: scripts/dashboard.py. Auto-refreshing loss curves, eval results + videos, experiment history, baseline comparison, status page. Accessible via Tailscale on 0.0.0.0:8501. |
| 2026-02-24 | Fast training mode: PushT task with Diffusion Policy (4.4M params, state-only, no video). Trains 1000 steps in ~4.5 min on CPU. `--task pusht` flag on train.py, evaluate.py, run_experiment.py. Dashboard updated for multi-task support. |

## Training Configurations

### PushT — Diffusion Policy (Fast)

- **Dataset:** `lerobot/pusht` — 206 demos, 25,650 frames at 10 FPS
- **Policy:** Diffusion Policy (DDPM) — 4.4M params
- **Observations:** 2D agent position (state-only, no images)
- **Actions:** 2D continuous (x, y movement)
- **Optimizer:** AdamW, lr=1e-4, cosine schedule, 500 warmup steps
- **Network:** 1D ConvUNet, down_dims=(64, 128, 256), horizon=16, n_action_steps=8
- **Training speed:** ~4.5 steps/s on CPU, ~4.5 min for 1000 steps

### Transfer Cube — ACT (Slow)

- **Dataset:** `lerobot/aloha_sim_transfer_cube_human` — 50 human demos, 20,000 frames at 50 FPS
- **Policy:** ACT (Action Chunking with Transformers) — 51.6M params
- **Observations:** RGB camera (480x640) + 14-DOF joint positions
- **Actions:** 14-DOF continuous (chunk size 100)
- **Optimizer:** AdamW, lr=1e-5, weight_decay=1e-4, grad_clip=10.0
- **Loss:** L1 action prediction + KL divergence (VAE, weight=10.0)
- **Training speed:** ~1.4 steps/s on MPS (Apple Silicon), ~60 min for 5000 steps

## Experiment Results

### PushT — 1000-step run

| Metric | Value |
|--------|------:|
| Loss (initial) | 1.34 |
| Loss (final) | 0.066 |
| Training Time | 4.5 min (CPU) |
| Parameters | 4.4M |
| Success Rate | 0% (0/10) |
| Avg Reward | 17.75 |

### Transfer Cube — 500-step run

| Metric | Ours (500 steps) | Pretrained (100k steps) |
|--------|----------------:|------------------------:|
| Loss (final) | 2.71 | — |
| Success Rate | 0% (0/5) | 80% (4/5) |
| Avg Reward | 0.00 | 179.80 |
| Training Time | 5.2 min | — |

## What's Next

- [x] Install LeRobot + dependencies (MuJoCo, Gymnasium)
- [x] Run a pretrained model in simulation
- [x] Evaluate and document results
- [x] Train a custom ACT policy from scratch
- [x] Evaluate trained checkpoints vs pretrained baseline
- [x] Build results visualization dashboard
- [x] Live Streamlit training dashboard (network-accessible via Tailscale)
- [x] Fast state-based training with PushT + Diffusion Policy
- [ ] Train PushT for more steps (10k+) to achieve task success
- [ ] Train ALOHA for more steps (50k-100k) to match pretrained performance
- [ ] Experiment with domain randomization
