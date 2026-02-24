# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## Status: Setting Up

**Current phase:** Environment setup — getting LeRobot + MuJoCo simulation running.

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

## Setup

```bash
git clone git@github.com:AviZurlo/robot-sim.git
cd robot-sim
# Setup instructions will be added as we build
```

## What's Next

- [ ] Install LeRobot + dependencies (MuJoCo, Gymnasium)
- [ ] Run a pretrained model in simulation
- [ ] Evaluate and document results
