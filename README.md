# VLA Diagnostic Probing

Comparing open-source vision-language-action (VLA) models using a suite of 8 diagnostic probes in MuJoCo simulation. Each probe tests a specific aspect of visual grounding, language conditioning, and spatial reasoning.

Inspired by [Avik De's probing framework](https://www.avikde.me/p/debugging-as-architecture-insight) for X-VLA.

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

## Probes

| # | Probe | What it tests |
|---|---|---|
| 1 | **Baseline** | Does the model reach for the correct object? Measures direction alignment and trajectory spread across seeds. |
| 2 | **Spatial Symmetry** | Does it track where things are? Swaps block positions — a grounded model re-routes. |
| 3 | **Camera Sensitivity** | Is it using the image? Shifts the camera laterally and measures output change. |
| 4 | **View Ablation** | Which camera does it depend on? Blacks out each view individually. |
| 5 | **Counterfactual** | Does it understand language or just match tokens? Tests synonym phrasings of the same instruction. |
| 6 | **Null Action** | Does "don't move" produce less motion? Tests instruction-following for negation/null commands. |
| 7 | **Attention** | Where is it looking? Extracts attention heatmaps and measures IoU overlap with target object pixels. |
| 8 | **Perturbation** | Does the trajectory adapt proportionally when the block moves? Tests spatial adaptation vs OOD collapse. |

## Models

| Model | Scene | Device | Notes |
|---|---|---|---|
| [X-VLA](https://github.com/TrossenRobotics/x-vla) | WidowX | MPS | Best spatial grounding |
| [OpenVLA](https://github.com/openvla/openvla) | WidowX | MPS | |
| [Pi0](https://github.com/Physical-Intelligence/openpi) | Franka | MPS | |
| [OpenVLA-OFT](https://github.com/openvla/openvla-oft) | Franka | MPS | |
| [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) | Franka | CUDA | RunPod A40 |

## Setup

```bash
git clone https://github.com/AviZurlo/robot-sim.git
cd robot-sim
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[xvla]"   # or openvla, pi0, openvla-oft
```

For Cosmos Policy (requires CUDA GPU):
```bash
# On RunPod — see scripts/run_cosmos_cloud.sh for full setup
bash scripts/setup_runpod.sh
```

## Running Probes

```bash
# Run all 8 probes for a model
python -m vla_probing.run_all --model xvla --scene widowx --device mps

# Run specific probes
python -m vla_probing.run_all --model pi0 --scene franka --probes baseline perturbation

# Interactive sim viewer (mjpython required)
.venv/bin/mjpython scripts/run_sim_interactive.py --scene widowx --model xvla

# Launch results dashboard
bash scripts/start_probe_dashboard.sh   # http://localhost:8502
```

Available models: `xvla`, `openvla`, `openvla_oft`, `pi0`, `cosmos_policy`

## Regenerating Results

```bash
python scripts/generate_results.py
```

Reads from `results/probes/`, writes figures to `results/figures/`, and updates the results section above.

## Repository Structure

```
vla_probing/
├── probes/          # 8 probe implementations
├── adapters/        # Model-specific inference wrappers
├── assets/          # MuJoCo scene XMLs (WidowX, Franka)
├── scene.py         # Scene management (reset, render, block manipulation)
├── adapter.py       # Base adapter interface
├── run_all.py       # Run all probes for a model
├── metrics.py       # Shared metric functions (DTW, perturbation sensitivity, IoU)
└── dashboard.py     # Streamlit results dashboard

scripts/
├── generate_results.py       # Regenerate README results section + figures
├── generate_viz.py           # Trajectory plots from NPZ artifacts
├── run_sim_interactive.py    # Live mjpython sim viewer with probe visualization
├── sim_viewer.py             # Browser-based Streamlit sim viewer
├── run_cosmos_cloud.sh       # Full RunPod setup + probe run for Cosmos Policy
├── setup_runpod.sh           # RunPod environment setup
└── start_probe_dashboard.sh  # Launch Streamlit dashboard

results/
├── probes/    # Probe result JSONs (one per model)
└── figures/   # Generated charts (radar + per-probe bar charts)

vendor/prismatic/   # Vendored Prismatic VLA components (used by OpenVLA-OFT adapter)
docs/               # Project notes and open questions
```
