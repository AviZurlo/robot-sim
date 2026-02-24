# Robot Sim

Run open source robot foundation models in simulation. Train policies, watch them improve with data.

## Status: Simulation Running

**Current phase:** Pretrained policy running in sim. Ready for data collection and training.

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

## Setup

```bash
git clone git@github.com:AviZurlo/robot-sim.git
cd robot-sim
# Setup instructions will be added as we build
```

## What's Next

- [x] Install LeRobot + dependencies (MuJoCo, Gymnasium)
- [x] Run a pretrained model in simulation
- [x] Evaluate and document results
- [ ] Collect simulation data and train a custom policy
- [ ] Try different policy architectures (Diffusion Policy, VQ-BeT)
- [ ] Experiment with domain randomization
