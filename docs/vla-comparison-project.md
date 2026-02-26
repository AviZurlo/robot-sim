# VLA Comparison Project — Debugging as Architecture Insight

**Inspired by:** [Avik De's article](https://www.avikde.me/p/debugging-as-architecture-insight) on probing X-VLA to understand what VLAs actually learn.

**Core idea:** Run the same diagnostic experiments across multiple VLA models to compare how different architectures understand vision, language, and action — and where they break.

**Hardware:** Mac mini M4, 24GB unified memory, PyTorch 2.7.1, MPS backend

---

## Goals

### Primary
1. **Reproduce Avik's probing suite** on X-VLA as a baseline
2. **Run the same probes on 3 additional VLAs** (π0, SmolVLA, OpenVLA) to compare architectural strengths/weaknesses
3. **Produce a structured comparison** with quantitative metrics and a robust results tracking framework

### Secondary
4. Understand which VLA architectures are most robust to real-world deployment concerns
5. Build reusable probing harness that can be applied to new VLAs as they come out
6. Blog post / content piece showing results (networking opportunity with Avik De, PI team, HuggingFace robotics)

### Non-Goals (for now)
- Closed-loop evaluation (running policies in full sim loops) — separate project
- Fine-tuning models (except where needed for fair embodiment matching)
- Speed/throughput benchmarking — this is about understanding, not performance

---

## Design Principles

1. **Zero-shot performance only** — use each model's best-supported embodiment from pretraining to normalize results
2. **Robust experiment tracking** — every probe result logged with full reproducibility metadata
3. **Leverage existing tools** — no rebuilding what's already available (LeRobot, LIBERO, Captum, W&B)
4. **All 4 models stay in scope** — X-VLA, π0, SmolVLA, OpenVLA
5. **AI agents do the implementation** — timelines are compressed vs. human execution

---

## The Probing Suite (from Avik's article)

| # | Probe | What it tests | Method |
|---|-------|--------------|--------|
| 1 | **Baseline trajectory** | Does the model reach for the right object? | Visualize action trajectory for "pick up the red block" |
| 2 | **Spatial symmetry** | Does the model understand absolute vs relative positions? | Swap block positions (red ↔ blue), compare trajectories |
| 3 | **Camera sensitivity** | Is spatial understanding tied to camera pose? | Mirror/rotate camera view, observe trajectory changes |
| 4 | **View ablation** | Which camera views matter? | Remove primary/secondary views one at a time |
| 5 | **Counterfactual prompts** | Does the language encoder understand synonyms? | "red block" vs "red cube" vs "crimson block" — should produce same action |
| 6 | **Null action prompts** | Does the model understand "don't move"? | "don't move" / "stay still" — should produce zero/minimal motion |
| 7 | **Attention visualization** | Where is the model looking? | Extract and overlay attention maps on input images |
| 8 | **Environment perturbation** | Does the model track object changes? | Move blocks to new positions, check trajectory adaptation |

---

## Model Selection & Embodiment Strategy

### Key Insight: Use Each Model's Native Embodiment

Instead of forcing all models onto WidowX (which creates unfair comparisons), we use each model's best-supported embodiment for zero-shot evaluation. The probes test *architectural behaviors*, not task-specific performance — so different embodiments are acceptable.

### Model Details

| # | Model | Params | Peak Memory (fp16) | Architecture | Best Embodiment | Checkpoint |
|---|-------|--------|-------------------|-------------|----------------|------------|
| **1** | **X-VLA** | 0.9B | ~2-4 GB | InternVL2 + soft prompts + flow matching | WidowX (primary training) | `lerobot/xvla-widowx` |
| **2** | **π0** | 3B | ~8-12 GB | PaliGemma + flow matching | LIBERO (Franka Panda) | `lerobot/pi0_libero_finetuned` |
| **3** | **SmolVLA** | ~0.5B | ~1-2 GB | SmolVLM2 + flow matching | SO-100 / community data | `lerobot/smolvla_base` |
| **4** | **OpenVLA-7B** | 7B | ~16-18 GB | Llama-2 + text-token actions | WidowX / BridgeV2 | `openvla/openvla-7b` |

### Embodiment Notes
- **X-VLA + OpenVLA** share WidowX — direct comparison possible on same embodiment
- **π0** has a LIBERO fine-tuned checkpoint — use LIBERO Franka Panda environment
- **SmolVLA** was trained on community robot data (SO-100) — most challenging for standard benchmarks; may need fine-tuning on LIBERO for fair comparison
- **Cross-embodiment probes** (same probe, different embodiment) are still valid for testing architectural properties like attention patterns, null action compliance, and semantic understanding

### Architecture Comparison

| Property | X-VLA | π0 | SmolVLA | OpenVLA |
|----------|-------|----|---------|---------|
| VLM backbone | InternVL2 (0.9B) | PaliGemma (3B) | SmolVLM2 (~0.5B) | Llama-2 + DINOv2 + SigLIP (7B) |
| Action head | Flow matching (10-step) | Flow matching | Flow matching | Autoregressive text tokens |
| Action space | Continuous | Continuous | Continuous | Discrete (text vocabulary) |
| Cross-embodiment | ✅ Soft prompts | ✅ Multi-embodiment | ❌ Needs fine-tuning | ❌ Needs fine-tuning |
| Stochastic actions | ✅ | ✅ | ✅ | ❌ (deterministic) |
| VLM queryable | ❌ | ❌ | ❌ | ✅ (actions = text tokens) |

### Architecture-Specific Probes

| Probe | X-VLA | π0 | SmolVLA | OpenVLA |
|-------|-------|-----|---------|---------|
| VLM scene description | ❌ | ❌ | ❌ | ✅ unique |
| Soft prompt inspection | ✅ unique | ❌ | ❌ | ❌ |
| Action stochasticity | ✅ | ✅ | ✅ | ❌ |
| Embodiment transfer | ✅ unique | ✅ | ❌ | ❌ |

---

## Tool Stack (Researched & Selected)

### Leverage existing tools — minimal custom code needed (~200-500 lines)

| Layer | Tool | Why |
|-------|------|-----|
| **Framework** | **LeRobot** | Natively supports X-VLA, π0, SmolVLA. Built-in eval infrastructure. |
| **Simulation** | **LIBERO** (primary) + **SimplerEnv** (WidowX) | Best model coverage. LIBERO has 4 task suites (spatial, object, goal, long) that map to our probes. |
| **Attention/Attribution** | **Captum** + `output_attentions=True` | Model-agnostic. ~50 lines per model wrapper. |
| **Trajectory Analysis** | **dtw-python** + **plotly** + **numpy** | DTW distance, L2 error, interactive 3D visualization. |
| **Experiment Tracking** | **Weights & Biases** | Best comparison UI. Image/trajectory artifacts. Side-by-side panels. Already supported by LeRobot. |
| **Reference Framework** | **RoboVLMs** | Most comprehensive existing VLA comparison (600+ experiments). Examine their eval code. |

### What We Build (Not Available Off-the-Shelf)
1. **Probe harness** — unified interface to run each probe across all 4 models
2. **Attention extraction wrappers** — per-model code to extract attention from the right layers (~50 lines each)
3. **Perturbation generators** — camera/prompt perturbation pipeline
4. **Metrics aggregation** — collect results into W&B tables for cross-model comparison
5. **Model loading wrapper** — thin adapter per model for consistent `predict_action()` API

---

## Evaluation Metrics

### Quantitative (tracked in W&B)

| # | Metric | What it measures | Applies to |
|---|--------|-----------------|------------|
| 1 | **L2 action error** | Euclidean distance between predicted and ground-truth actions (from dataset) | All models |
| 2 | **Trajectory spread** | Action variance across 10+ random seeds | Flow matching models (X-VLA, π0, SmolVLA) |
| 3 | **Perturbation sensitivity** | L2 delta in predicted action per variable change (prompt, camera, object position) | All models |
| 4 | **Attention IoU** | Overlap between attention map and ground-truth object region | All models |
| 5 | **Trajectory DTW** | Dynamic Time Warping distance between predicted and reference trajectories | All models |
| 6 | **Trajectory smoothness** | Mean absolute jerk (3rd derivative of position) | All models |

### Qualitative (logged as W&B artifacts)
- Trajectory visualization (side-by-side per model per probe)
- Attention map overlays on input images
- Failure mode taxonomy (wrong object, wrong direction, random/noisy, frozen/minimal)

### Results Tracking Schema

Each experiment logged to W&B with:
```
{
  model: str,           # "xvla" | "pi0" | "smolvla" | "openvla"
  embodiment: str,      # "widowx" | "libero_franka" | "so100"
  probe: str,           # "baseline" | "spatial_symmetry" | "camera_sensitivity" | ...
  probe_variant: str,   # specific variant (e.g., "mirror_horizontal", "prompt_synonym_cube")
  seed: int,
  metrics: {
    l2_error: float,
    trajectory_dtw: float,
    trajectory_jerk: float,
    attention_iou: float,
  },
  artifacts: {
    trajectory_plot: Image,
    attention_map: Image,
    raw_actions: Array,
  }
}
```

---

## Execution Plan

### Phase 1: X-VLA Baseline
**Goal:** Reproduce Avik's probes locally with quantitative metrics

- Fork [avikde/vla-pipeline](https://github.com/avikde/vla-pipeline)
- Adapt for MPS (swap CUDA → MPS, force `attn_implementation="eager"` for InternVL2)
- CPU fallback if MPS is unstable — don't let MPS issues block research
- Set up W&B project, define logging schema
- Run all 8 probes on X-VLA WidowX
- Add quantitative metrics (L2 error, trajectory spread, attention IoU)
- Validate results match Avik's findings

### Phase 2: Probing Harness + π0
**Goal:** Abstract probes into reusable harness, add second model

- Extract probe logic into model-agnostic functions
- Define `VLAAdapter` interface (load, predict, get_attention)
- Implement π0 adapter using `lerobot/pi0_libero_finetuned`
- Set up LIBERO environment for π0 evaluation
- Run full probe suite on π0
- First cross-model comparison (X-VLA vs π0)

### Phase 3: SmolVLA + OpenVLA
**Goal:** Complete the 4-model comparison

- SmolVLA adapter (smallest model, fast iteration)
- OpenVLA adapter (fp16, careful memory management)
- OpenVLA-specific probe: VLM scene querying ("what do you see?")
- Run full probe suite on both
- Cross-model comparison with all 4

### Phase 4: Analysis & Content
**Goal:** Produce publishable comparison

- Side-by-side visualizations (probe × model grid)
- Quantitative results tables from W&B
- Architectural insights: what do different designs actually buy you?
- Deployment implications: which failure modes matter for real robots?
- Blog post draft
- Share with Avik De, PI team, HuggingFace robotics team

---

## Key Research Questions

1. **Do different action representations (continuous flow matching vs text tokens) produce fundamentally different failure modes?**
2. **Does model scale (0.5B → 7B) improve spatial understanding, or just task coverage?**
3. **Are soft prompts (X-VLA) actually encoding embodiment-specific spatial reasoning, or just biases?**
4. **Does any architecture show genuine null-action compliance (understanding "don't move")?**
5. **How much of VLA behavior is spatial template matching (training distribution) vs actual scene understanding?**

---

## Prior Art

- **[RoboVLMs](https://robovlms.github.io)** — 600+ experiment VLA comparison framework (task success focus, not interpretability)
- **[Avik De's VLA Pipeline](https://github.com/avikde/vla-pipeline)** — X-VLA probing notebook + MuJoCo WidowX scene
- **[SimplerEnv](https://simpler-env.github.io/)** — Real-to-sim evaluation framework (CoRL 2024)
- **[X-VLA paper](https://arxiv.org/abs/2510.10274)** — Compares X-VLA against OpenVLA, Octo on SimplerEnv + CALVIN + LIBERO
- **[SmolVLA paper](https://arxiv.org/abs/2506.01844)** — Efficient VLA design for community hardware

**What's novel about our work:** Nobody has done systematic *interpretability probing* across VLA architectures. Existing comparisons measure task success; we measure *what the models understand*.

---

## Relationship to Existing Robot-Sim Project

| | Robot-Sim (existing) | VLA Comparison (new) |
|---|---|---|
| **Focus** | Training policies from scratch | Understanding pretrained VLA behavior |
| **Models** | ACT, Diffusion Policy | X-VLA, π0, SmolVLA, OpenVLA |
| **Loop** | Closed-loop (train → evaluate in sim) | Open-loop (single-step probing) |
| **Shared** | Same Mac mini, same PyTorch/lerobot stack |

VLA comparison work goes in `vla-probing/` subdirectory of the robot-sim repo.

---

*v3 — 2026-02-26 (updated with council review, tools research, embodiment strategy)*
