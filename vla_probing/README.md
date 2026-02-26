# VLA Probing Suite

Diagnostic probes for Vision-Language-Action (VLA) models, inspired by [Avik De's "Debugging as Architecture Insight"](https://www.avikde.me/p/debugging-as-architecture-insight).

## Overview

This suite runs 8 diagnostic probes against VLA models to understand what they actually learn about vision, language, and action — beyond task success rates.

**Current models:** X-VLA (0.9B, WidowX)
**Hardware:** Mac mini M4, 24GB unified memory, MPS backend

## Setup

```bash
# From the robot-sim root directory
uv sync --extra vla-probing

# This installs lerobot[xvla], captum, plotly, wandb, dtw-python, scipy
```

The X-VLA model weights (~1.8GB fp16) will be downloaded automatically from HuggingFace on first run.

## Running Probes

### Run all probes
```bash
python -m vla_probing --model xvla --device mps
```

### Run a specific probe
```bash
python -m vla_probing baseline --model xvla --device mps
python -m vla_probing spatial_symmetry
python -m vla_probing attention --wandb  # with W&B logging
python -m vla_probing counterfactual --group spatial_primitives
```

### With W&B experiment tracking
```bash
wandb login  # one-time setup
python -m vla_probing --model xvla --wandb
```

### CPU fallback (if MPS has issues)
```bash
python -m vla_probing --model xvla --device cpu
```

## The 8 Probes

| # | Probe | What it tests | Module |
|---|-------|--------------|--------|
| 1 | **Baseline** | Does the model reach for the right object? | `probes/baseline.py` |
| 2 | **Spatial Symmetry** | Absolute vs relative position understanding | `probes/spatial_symmetry.py` |
| 3 | **Camera Sensitivity** | Is spatial reasoning tied to camera pose? | `probes/camera_sensitivity.py` |
| 4 | **View Ablation** | Which camera views matter most? | `probes/view_ablation.py` |
| 5 | **Counterfactual** | Does the language encoder understand synonyms? | `probes/counterfactual.py` |
| 6 | **Null Action** | Does the model understand "don't move"? | `probes/null_action.py` |
| 7 | **Attention** | Where is the model looking? | `probes/attention.py` |
| 8 | **Perturbation** | Does the model track object changes? | `probes/perturbation.py` |

## Metrics

Each probe computes a subset of these quantitative metrics:

- **L2 action error** — distance to ground-truth actions (when available)
- **Trajectory DTW** — Dynamic Time Warping distance between trajectories
- **Trajectory jerk** — smoothness (3rd derivative of position)
- **Trajectory spread** — action variance across random seeds (flow matching stochasticity)
- **Attention IoU** — overlap between attention map and ground-truth object region
- **Perturbation sensitivity** — L2 delta per variable change

Results are saved to `outputs/probes/probe_results_<model>.json` and optionally logged to W&B.

## Architecture

```
vla_probing/
├── adapter.py          # VLAAdapter interface + XVLAAdapter
├── scene.py            # MuJoCo WidowX scene renderer
├── metrics.py          # Quantitative metrics (L2, DTW, jerk, IoU)
├── tracking.py         # W&B experiment tracking
├── run_all.py          # Full probe suite runner
├── __main__.py         # CLI entry point
├── probes/
│   ├── base.py         # Base probe class
│   ├── baseline.py
│   ├── spatial_symmetry.py
│   ├── camera_sensitivity.py
│   ├── view_ablation.py
│   ├── counterfactual.py
│   ├── null_action.py
│   ├── attention.py
│   └── perturbation.py
└── assets/
    └── widowx/         # MuJoCo WidowX scene (from avikde/vla-pipeline)
        ├── wx250s.xml
        ├── widowx_vision_scene.xml
        └── assets/     # STL meshes + textures
```

## Adding New Models

Implement the `VLAAdapter` interface:

```python
from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput

class MyModelAdapter(VLAAdapter):
    model_name = "my_model"

    def load_model(self, device: str = "mps") -> None: ...
    def predict_action(self, inp: VLAInput) -> VLAOutput: ...
    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]: ...

    @property
    def action_dim(self) -> int: ...
    @property
    def chunk_size(self) -> int: ...
```

Then register it in `probes/base.py:make_adapter()`.

## MPS Notes

- X-VLA uses `attn_implementation='eager'` (flash_attn not available on MPS)
- If specific ops fail on MPS, use `--device cpu` as fallback
- Model is 0.9B params (~1.8GB fp16) — fits easily in 24GB unified memory
