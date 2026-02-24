# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## Status: Results Dashboard Live

**Current phase:** Visualization pipeline built. Training loss curves, eval metrics, and baseline comparisons auto-generated as PNG plots.

## Architecture

- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face)
- **Simulation:** Gymnasium + MuJoCo
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

## Training Details

- **Dataset:** `lerobot/aloha_sim_transfer_cube_human` — 50 human demos, 20,000 frames at 50 FPS
- **Policy:** ACT (Action Chunking with Transformers) — 51.6M params
- **Observations:** RGB camera (480x640) + 14-DOF joint positions
- **Actions:** 14-DOF continuous (chunk size 100)
- **Optimizer:** AdamW, lr=1e-5, weight_decay=1e-4, grad_clip=10.0
- **Loss:** L1 action prediction + KL divergence (VAE, weight=10.0)
- **Training speed:** ~1.4 steps/s on MPS (Apple Silicon), ~60 min for 5000 steps

## Experiment Results (500-step run)

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
- [ ] Train for more steps (50k-100k) to match pretrained performance
- [ ] Try different policy architectures (Diffusion Policy, VQ-BeT)
- [ ] Experiment with domain randomization
