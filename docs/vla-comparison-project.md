# VLA Comparison Project — Debugging as Architecture Insight

**Inspired by:** [Avik De's article](https://www.avikde.me/p/debugging-as-architecture-insight) on probing X-VLA to understand what VLAs actually learn.

**Core idea:** Run the same diagnostic experiments across multiple VLA models to compare how different architectures understand vision, language, and action — and where they break.

**Hardware:** Mac mini M4, 24GB unified memory, PyTorch 2.7.1, MPS backend

---

## Goals

### Primary
1. ✅ **Reproduce Avik's probing suite** on X-VLA as a baseline
2. ✅ **Run the same probes on additional VLAs** (π0, OpenVLA) to compare architectural strengths/weaknesses
3. ✅ **Produce a structured comparison** with quantitative metrics, interactive dashboard, and robust results tracking
4. **Test each model on its native embodiment** for fair comparison

### Secondary
5. Understand which VLA architectures are most robust to real-world deployment concerns
6. Build reusable probing harness that can be applied to new VLAs as they come out
7. Blog post / content piece showing results (networking opportunity with Avik De, PI team, HuggingFace robotics)

### Future Expansion
8. **GR00T N1.6** — NVIDIA's 3B VLA (Cosmos-Reason + DiT action head). Weights available, needs CUDA or eager-attention workaround.
9. **mimic-video** — Video-Action Model (Cosmos video backbone + flow matching action decoder). No public weights yet.
10. **DreamDojo** — World model approach. Could be probed once paired with an action decoder.
11. More video-model-based policies as they emerge

### Non-Goals (for now)
- Closed-loop evaluation (running policies in full sim loops) — separate project
- Fine-tuning models (except where needed for fair embodiment matching)
- Speed/throughput benchmarking — this is about understanding, not performance

---

## Design Principles

1. **Zero-shot performance only** — use each model's best-supported embodiment from pretraining
2. **Native embodiment testing** — each model tested on the embodiment it was trained on
3. **Robust experiment tracking** — every probe result logged with full reproducibility metadata
4. **Leverage existing tools** — no rebuilding what's already available (LeRobot, MuJoCo, W&B)
5. **Black-box probing** — probes work regardless of internal architecture (flow matching, autoregressive, video prediction)
6. **AI agents do the implementation** — timelines are compressed vs. human execution

---

## Models

### Active Models (with results)

| # | Model | Params | Architecture | Action Space | Embodiment(s) | Status |
|---|-------|--------|-------------|-------------|---------------|--------|
| **1** | **X-VLA** | 0.9B | InternVL2 + soft prompts + flow matching | Continuous 20D | WidowX (native) | ✅ Complete |
| **2** | **π0** | 3B | PaliGemma + Gemma action expert + flow matching | Continuous 7D | Franka (native) + WidowX (cross) | ✅ Complete |
| **3** | **OpenVLA** | 7B | Llama-2 + DINOv2 + SigLIP → discrete tokens | Discrete 7D (256 bins) | WidowX (native) | ✅ Complete |

### Removed
- **SmolVLA** (0.5B) — Removed. No standard simulation for SO-100 training embodiment. Cross-embodiment results didn't add signal.

### Future Models
- **GR00T N1.6** (3B) — NVIDIA VLA. Cosmos-Reason-2B VLM + DiT. Public weights. Needs CUDA (possible eager-attention workaround). Has BridgeV2 + LIBERO finetuned checkpoints.
- **mimic-video** — Video-Action Model. Cosmos video backbone + flow matching IDM. No public weights yet.
- **DreamDojo** — World model (not a VLA). Would need different probe design or pairing with action decoder.

### Architecture Comparison

| Property | X-VLA | π0 | OpenVLA |
|----------|-------|----|---------|
| VLM backbone | InternVL2 (0.9B) | PaliGemma (3B) | Llama-2 + DINOv2 + SigLIP (7B) |
| Action head | Flow matching (10-step) | Flow matching (10-step) | Autoregressive text tokens |
| Action space | Continuous | Continuous | Discrete (256 bins/dim) |
| Variance mechanism | Random seed (starting noise) | Random seed (starting noise) | Sampling temperature |
| Temperature/seed | Seeds 0-9 | Seeds 0-9 | **T=0.5** |
| Camera inputs | 2 (primary + secondary) | 2 (agentview + wrist) | 1 (single camera) |
| Proprioception | Yes (8D state) | Yes (8D state) | No |
| VLM queryable | ❌ | ❌ | ✅ (actions = text tokens) |

---

## The Probing Suite

### 8 Diagnostic Probes

| # | Probe | What It Tests | Key Metric | Good Direction |
|---|-------|--------------|------------|----------------|
| 1 | **Baseline Trajectory** | Does the model reach for the correct object? | `direction_alignment` | ↑ Higher is better |
| 2 | **Spatial Symmetry** | Does the model understand absolute vs relative positions? | `perturbation_sensitivity` | ↑ Higher is better |
| 3 | **Camera Sensitivity** | Is spatial reasoning tied to camera pose? | `mirror_camera_sensitivity` | ↑ Higher is better |
| 4 | **View Ablation** | Which camera views carry the most information? | `full_vision_ablation_sensitivity` | ↑ Higher is better |
| 5 | **Counterfactual Prompts** | Does the language encoder collapse synonyms correctly? | `mean_synonym_sensitivity` | ↓ Lower is better |
| 6 | **Null Action** | Can the model comply with "don't move"? | `null_vs_baseline_ratio` | ↓ Lower is better (0 = stays still) |
| 7 | **Attention Visualization** | Is the model attending to the referenced object? | `mean_attention_iou` | ↑ Higher is better |
| 8 | **Environment Perturbation** | Does the model re-plan when objects move? | `mean_perturbation_sensitivity` | ↑ Higher is better |

### Variance Methodology

**Flow matching models** (X-VLA, π0): Tested with multiple random seeds (default 10). Different seeds produce different starting noise for the denoising process.

**Autoregressive models** (OpenVLA): Uses **sampling temperature = 0.5** to introduce controlled randomness in token selection. Temperature 0 would be fully deterministic.

See dashboard About page for full educational explainer.

---

## Results Summary

### Native Embodiment Results

| Model | Embodiment | Direction Alignment | Spatial Sensitivity | Camera Sensitivity | Null Action Ratio | Attention IoU |
|-------|-----------|-------------------|--------------------|--------------------|-------------------|---------------|
| **X-VLA** | WidowX (native) | **0.87** | 0.002 | 0.056 | 1.0 | 0.09 |
| **π0** | Franka (native) | **0.47** | 1.66 | 1.66 | ~1.0 | 0.0 |
| **π0** | WidowX (cross) | -0.01 | 1.74 | 1.74 | ~1.0 | 0.0 |
| **OpenVLA** | WidowX (native) | N/A* | 0.145 | 0.145 | ~1.0 | 0.11 |

*OpenVLA outputs single-step actions (chunk_size=1) with zero jerk — direction alignment metric not directly comparable.

### Key Findings

1. **Embodiment match matters enormously**: π0 went from -0.01 (random) on WidowX to 0.47 (purposeful) on native Franka. This is the strongest result in the project.

2. **No model shows null-action compliance**: All models produce similar displacement whether told "pick up the red block" or "don't move." None understand negation.

3. **OpenVLA quantization dead zone**: The 256-bin discretization creates insensitivity to small perturbations. Synonym sensitivity = 0 (identical tokens for different phrasings). Block position shifts produce identical actions. This is architectural, not a bug.

4. **Attention extraction is model-dependent**: Works for X-VLA (IoU 0.09) and OpenVLA (IoU 0.11), fails for π0 (returns zeros). Different architectures require different extraction strategies.

5. **X-VLA has the best direction alignment** (0.87) but diffuse attention — it reaches for the right thing without clearly "looking" at it.

### Experiment Audit

Full methodology audit at `docs/experiment-audit.md`. Confidence levels:
- **X-VLA**: High confidence — native embodiment, no workarounds
- **π0**: Medium-high confidence — native Franka scene works well, some compatibility patches needed
- **OpenVLA**: Medium confidence — temperature fix resolved determinism, but quantization effects dominate

---

## Scenes

### MuJoCo Scenes Built

| Scene | Robot | Cameras | State | Assets |
|-------|-------|---------|-------|--------|
| **WidowX** | WidowX 250s | "up" (over-shoulder) + "side" | 8D BridgeData format | `vla_probing/assets/widowx/` |
| **Franka** | Franka Emika Panda | "agentview" (front) + "robot0_eye_in_hand" (wrist) | 8D LIBERO format | `vla_probing/assets/franka/` (MuJoCo Menagerie) |

Both scenes include: table, red block, blue block, manipulable positions, matching camera conventions for their respective training datasets.

---

## Tool Stack

| Layer | Tool | Status |
|-------|------|--------|
| **Framework** | LeRobot | ✅ Active |
| **Simulation** | MuJoCo (WidowX + Franka scenes) | ✅ Active |
| **Dashboard** | Streamlit (port 8502) | ✅ Active |
| **Attention** | Model-specific hooks + Captum | ✅ Partial (fails for π0) |
| **Trajectory Analysis** | dtw-python + plotly + numpy | ✅ Active |
| **Experiment Tracking** | W&B (optional) + JSON results | ✅ Active |

---

## Project Structure

```
vla_probing/
├── adapter.py              # VLAAdapter interface
├── adapters/
│   ├── __init__.py
│   ├── openvla.py          # OpenVLA-7B adapter (T=0.5 sampling)
│   └── pi0.py              # π0 adapter (KV cache patch, SigLIP shim)
├── assets/
│   ├── widowx/             # WidowX MuJoCo scene
│   └── franka/             # Franka Panda MuJoCo scene (from Menagerie)
├── probes/
│   ├── base.py             # Probe base class + factories
│   ├── baseline.py         # Probe 1: Baseline trajectory
│   ├── spatial_symmetry.py # Probe 2: Block position swap
│   ├── camera_sensitivity.py # Probe 3: Camera mirror/flip
│   ├── view_ablation.py    # Probe 4: Remove camera views
│   ├── counterfactual.py   # Probe 5: Synonym prompts
│   ├── null_action.py      # Probe 6: "Don't move" compliance
│   ├── attention.py        # Probe 7: Attention map extraction
│   ├── perturbation.py     # Probe 8: Block position shifts
│   └── vlm_query.py        # Probe 9: VLM scene description (OpenVLA only)
├── scene.py                # WidowXScene + FrankaScene classes
├── metrics.py              # Shared metric computation
├── tracking.py             # W&B experiment tracking
├── dashboard.py            # Streamlit dashboard
├── run_all.py              # Full suite runner
└── __main__.py             # CLI entry point

outputs/probes/
├── probe_results_xvla_widowx.json
├── probe_results_pi0_franka.json
├── probe_results_pi0_widowx.json
└── probe_results_openvla_widowx.json

docs/
├── vla-comparison-project.md   # This file
├── experiment-audit.md         # Methodology audit report
├── open-questions-research.md  # Research notes
└── tools-research.md           # Tool selection research
```

---

## Execution History

### Phase 1: X-VLA Baseline ✅
- Forked avikde/vla-pipeline, adapted for MPS
- Built `vla_probing/` package with VLAAdapter interface
- All 8 probes implemented and running
- First results: direction alignment 0.87, diffuse attention (IoU 0.09)
- Merged to main (commit `5c3ea53`)

### Phase 2: Multi-Model Expansion ✅
- π0 adapter with KV cache fix, attention mask patch, SigLIP shim
- OpenVLA adapter with fp16 on MPS, VLM querying probe
- SmolVLA adapter (later removed — no native sim environment)
- All adapters tested with 8 probes on WidowX scene
- Streamlit dashboard built (port 8502)

### Phase 3: Native Embodiment Testing ✅
- Downloaded Franka Panda from MuJoCo Menagerie
- Built FrankaScene class with LIBERO-style cameras
- Made probe runner scene-aware (`--scene widowx|franka`)
- π0 on Franka: direction alignment jumped from -0.01 to 0.47
- Removed SmolVLA (no SO-100 sim available)

### Phase 4: Methodology Validation ✅
- Spawned audit agent — 14K-word methodology report
- Fixed OpenVLA determinism (do_sample=True, temperature=0.5)
- Added per-probe color normalization to dashboard (green=good, red=bad)
- Added educational variance methodology explainer
- Removed duplicate result files, added embodiment labels

### Phase 5: Analysis & Content (Next)
- [ ] Cross-model comparison analysis with key insights
- [ ] Blog post / writeup
- [ ] Share with Avik De
- [ ] Clean up old agent tmux sessions (6 idle)

### Phase 6: Future Models (Planned)
- [ ] GR00T N1.6 — try eager-attention on CPU/MPS, or use cloud GPU
- [ ] mimic-video — when weights are released
- [ ] Increase seed count (10 → 50) for more robust variance estimates
- [ ] Add repeated trials per probe variant for confidence intervals

---

## Key Research Questions

1. ✅ **Do different action representations produce fundamentally different failure modes?** — Yes. Flow matching models show stochastic variation; OpenVLA's discrete tokens create quantization dead zones. Fundamentally different failure signatures.

2. 🔄 **Does model scale (0.9B → 7B) improve spatial understanding?** — Not clearly. X-VLA (0.9B) outperforms OpenVLA (7B) on direction alignment. Scale alone doesn't buy spatial understanding.

3. 🔄 **Are soft prompts encoding embodiment-specific spatial reasoning?** — Partially supported. X-VLA's strong WidowX performance vs weak cross-embodiment transfer suggests embodiment encoding.

4. ✅ **Does any architecture show genuine null-action compliance?** — No. All three models produce similar displacement regardless of "don't move" instructions. None understand negation.

5. ✅ **How much is spatial template matching vs actual understanding?** — Mostly template matching. π0 goes from random to purposeful just by matching the visual domain. Models don't generalize spatial reasoning across embodiments.

---

## References

- [Avik De — Debugging as Architecture Insight](https://www.avikde.me/p/debugging-as-architecture-insight)
- [Avik De — The Architecture Behind "End-to-End" Robotics Pipelines](https://www.avikde.me/p/the-architecture-behind-end-to-end)
- [X-VLA Paper (arXiv 2510.10274)](https://arxiv.org/abs/2510.10274)
- [avikde/vla-pipeline (GitHub)](https://github.com/avikde/vla-pipeline)
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
- [GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA's foundation VLA
- [mimic-video](https://mimic-video.github.io/) — Video-Action Model (arXiv 2512.15692)
- [DreamDojo](https://github.com/NVIDIA/DreamDojo) — Interactive world model (arXiv 2602.06949)
- [RoboVLMs](https://robovlms.github.io/) — VLA comparison framework (task success focus)

---

*v4 — 2026-02-26 (updated with all completed phases, native embodiment results, methodology audit, future model roadmap)*
