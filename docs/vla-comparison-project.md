# VLA Comparison Project — Debugging as Architecture Insight

**Inspired by:** [Avik De's article](https://www.avikde.me/p/debugging-as-architecture-insight) on probing X-VLA to understand what VLAs actually learn.

**Core idea:** Run the same diagnostic experiments across multiple VLA models to compare how different architectures understand vision, language, and action — and where they break.

**Hardware:** Mac mini M4, 24GB unified memory, PyTorch 2.7.1, MPS backend  
**Cloud GPU:** RunPod A40 48GB for CUDA-only models (Cosmos Policy, GR00T)

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
| **4** | **OpenVLA-OFT** | 7B | Same Prismatic VLM + continuous action head (OFT fine-tuned) | Continuous 7D | Franka (native) | ✅ Complete |
| **5** | **Cosmos Policy** | 2B | Cosmos video diffusion + T5-XXL text + DiT | Continuous 16-step chunks + future images + values | Franka (LIBERO) | 🔄 RunPod setup in progress |
| **6** | **GR00T N1.6** | 3B | Cosmos-Reason-2B VLM + DiT action head | Continuous 7D | Franka (LIBERO/BridgeV2) | 📋 Adapter written, queued after Cosmos |

### Removed
- **SmolVLA** (0.5B) — Removed. No standard simulation for SO-100 training embodiment. Cross-embodiment results didn't add signal.

### Future Models
- **GR00T N1.6** (3B) — NVIDIA VLA. Cosmos-Reason-2B VLM + DiT. Public weights. Needs CUDA (possible eager-attention workaround). Has BridgeV2 + LIBERO finetuned checkpoints.
- **mimic-video** — Video-Action Model. Cosmos video backbone + flow matching IDM. No public weights yet.
- **DreamDojo** — World model (not a VLA). Would need different probe design or pairing with action decoder.

### Architecture Comparison

| Property | X-VLA | π0 | OpenVLA | OpenVLA-OFT | Cosmos Policy |
|----------|-------|----|---------|-------------|---------------|
| VLM backbone | InternVL2 (0.9B) | PaliGemma (3B) | Llama-2 + DINOv2 + SigLIP (7B) | Same Prismatic VLM (7B) | Cosmos video encoder + T5-XXL |
| Action head | Flow matching (10-step) | Flow matching (10-step) | Autoregressive text tokens | Continuous regression (OFT) | Video diffusion DiT |
| Action space | Continuous | Continuous | Discrete (256 bins/dim) | Continuous | Continuous (16-step chunks) |
| Variance mechanism | Random seed (starting noise) | Random seed (starting noise) | Sampling temperature | Deterministic | Random seed (diffusion noise) |
| Temperature/seed | Seeds 0-9 | Seeds 0-9 | **T=0.5** | N/A (deterministic) | Seeds 0-9 |
| Camera inputs | 2 (primary + secondary) | 2 (agentview + wrist) | 1 (single camera) | 1 (single camera) | 2 (agentview + wrist) |
| Proprioception | Yes (8D state) | Yes (8D state) | No | No | Yes (LIBERO format) |
| VLM queryable | ❌ | ❌ | ✅ (actions = text tokens) | ✅ (same VLM) | ❌ |
| Outputs | Actions only | Actions only | Actions only | Actions only | Actions + future images + values |

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
| **OpenVLA-OFT** | Franka (native) | **0.53** | 0.00008 | 0.0 | 1.04 | N/A |
| **Cosmos Policy** | Franka (LIBERO) | 0.23 | 0.0 | 0.0 | 1.10 | N/A† |

*OpenVLA outputs single-step actions (chunk_size=1) with zero jerk — direction alignment metric not directly comparable.
†Cosmos Policy uses a video diffusion DiT, not a standard VLM — attention extraction requires different approach.

### Key Findings

1. **Embodiment match matters enormously**: π0 went from -0.01 (random) on WidowX to 0.47 (purposeful) on native Franka. This is the strongest result in the project.

2. **No model shows null-action compliance**: All models produce similar displacement whether told "pick up the red block" or "don't move." None understand negation.

3. **OpenVLA quantization dead zone**: The 256-bin discretization creates insensitivity to small perturbations. Synonym sensitivity = 0 (identical tokens for different phrasings). Block position shifts produce identical actions. This is architectural, not a bug.

4. **VLM dead zone is backbone-level, not action-head-level**: OpenVLA-OFT replaces discrete tokens with continuous regression but keeps the same Prismatic VLM. Result: direction alignment improves dramatically (0.08→0.53), but perturbation sensitivity stays near zero (0.00008). **The VLM backbone itself is insensitive to scene perturbations — discretization is not the root cause.**

5. **Attention extraction is model-dependent**: Works for X-VLA (IoU 0.09) and OpenVLA (IoU 0.11), fails for π0 (returns zeros). Different architectures require different extraction strategies.

6. **X-VLA has the best direction alignment** (0.87) but diffuse attention — it reaches for the right thing without clearly "looking" at it.

7. **OFT fixes action quality but not perception**: OFT's continuous actions produce better direction alignment (0.53 vs base OpenVLA's quantized outputs) and non-zero synonym sensitivity (0.047), but the Prismatic VLM backbone remains a perception bottleneck.

8. **Cosmos Policy: proprioceptive distribution fragility**: The video-diffusion architecture produces a fixed trajectory largely independent of visual input, language commands, or spatial perturbations. Root cause: Cosmos Policy normalizes proprioception using LIBERO dataset statistics, and our standard Franka home configuration (`[0, 0, 0, -1.57, 0, 1.57, -0.785, 0.04, 0.04]`) falls massively outside LIBERO's narrow joint ranges (joints 3,4,5 normalize to -4.4, -1.0, +2.6 respectively). The model collapses to a default trajectory when proprio is OOD. **Critically, π0 handles the same joint configuration gracefully (0.47 direction alignment), exposing Cosmos Policy's brittleness as model-specific, not a test artifact.** This is a real deployment concern: any Franka not starting in the exact LIBERO training configuration will produce meaningless actions. The zero perturbation sensitivity (0.0 across all perturbations) and near-unity null-action ratio (1.10) are consequences of this collapse — the model is in a degenerate regime where inputs don't influence outputs.

9. **View ablation reveals partial visual processing in Cosmos**: Despite the proprio-induced collapse, zeroing out the wrist camera changes the trajectory (secondary ablation sensitivity = 0.46), suggesting the video encoder still processes visual input even when proprio is OOD. The primary camera has minimal impact (0.07), consistent with the wrist view being more informative in LIBERO's close-range manipulation tasks.

### Experiment Audit

Full methodology audit at `docs/experiment-audit.md`. Confidence levels:
- **X-VLA**: High confidence — native embodiment, no workarounds
- **π0**: Medium-high confidence — native Franka scene works well, some compatibility patches needed
- **OpenVLA**: Medium confidence — temperature fix resolved determinism, but quantization effects dominate
- **Cosmos Policy**: Medium confidence — model runs correctly on CUDA, but proprioception is out of LIBERO's training distribution (standard Franka home qpos ≠ LIBERO home qpos). Results reflect genuine deployment-scenario behavior (what happens when you run a LIBERO-trained model on a standard Franka config), but cannot be interpreted as "what Cosmos Policy does in-distribution." The zero-sensitivity results are a distribution mismatch finding, not a measure of the model's peak capability.

---

## Scenes

### MuJoCo Scenes Built

| Scene | Robot | Cameras | State | Assets |
|-------|-------|---------|-------|--------|
| **WidowX** | WidowX 250s | "up" (over-shoulder) + "side" | 8D BridgeData format | `vla_probing/assets/widowx/` |
| **Franka** | Franka Emika Panda | "agentview" (front) + "robot0_eye_in_hand" (wrist) | 8D LIBERO format | `vla_probing/assets/franka/` (MuJoCo Menagerie) |

Both scenes include: table, red block, blue block, manipulable positions, matching camera conventions for their respective training datasets.

---

## Cloud GPU Infrastructure (RunPod)

**Why:** Cosmos Policy and GR00T N1.6 require CUDA (flash attention, transformer engine). Mac MPS backend unsupported.

**Setup:**
- **Provider:** RunPod ($0.79/hr for A40 48GB)
- **Pod:** `useless_aquamarine_moose` (ID: `rzcq7u77xyg2yj`) — A40 48GB GPU, 200GB container disk
- **SSH:** `ssh -tt rzcq7u77xyg2yj-64411be2@ssh.runpod.io -i ~/.ssh/id_ed25519` (requires PTY)
- **Workspace:** `/workspace/robot-sim/` (repo cloned, `feat/cosmos-policy-adapter` branch)
- **Model cache:** Cosmos-Policy-LIBERO-Predict2-2B + T5-11b downloaded to default HF cache

**Setup script:** `scripts/setup_runpod.sh` — one-command bootstrap for new pods:
```bash
# From RunPod terminal:
git clone git@github.com:AviZurlo/robot-sim.git /workspace/robot-sim
cd /workspace/robot-sim && bash scripts/setup_runpod.sh
```

**Minimal deps:** `requirements-runpod.txt` — only what probes need (mujoco, pillow, plotly, dtw-python, scipy, wandb). No lerobot, streamlit, or Mac-specific packages.

**Import patches applied** (Cosmos Policy has hard deps on libero, transformer_engine, flash_attn):
- `libero` imports wrapped in try/except in `run_libero_eval.py` and `libero_utils.py`
- `transformer_engine` / `transformer_engine_torch` imports wrapped across 5+ files
- `flash_attn_2` availability assert bypassed in `qwen2_5_vl.py`
- `apply_rotary_pos_emb` fallback added in `minimal_v4_dit.py`
- Missing deps installed: `draccus`, `pytz`

**Environment for running probes:**
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_TOKEN=$HF_TOKEN
export PYTHONPATH=/workspace/cosmos-policy:/workspace/robot-sim
cd /workspace/robot-sim && python -m vla_probing --model cosmos_policy --scene franka --device cuda
```

**Cost management:** Pod is pay-per-hour. Stop when not actively running probes.

**Lessons learned:**
- **Container disk must be ≥200GB** — T5-11b alone is 85GB, plus Cosmos checkpoint, flash-attn build artifacts, system packages
- Default 20GB root disk causes repeated "No space left on device" failures
- `transformer-engine[core_cu12]` installs prebuilt core, but `transformer_engine_torch` must be built from source with cuDNN headers
- Symlink cuDNN headers from pip nvidia package: `ln -sf /usr/local/lib/python3.11/dist-packages/nvidia/cudnn/include/*.h /usr/include/`
- `MUJOCO_GL=egl` + `libegl1-mesa-dev` required for headless MuJoCo rendering
- RunPod SSH proxy requires PTY (`-tt` flag); direct TCP port also works when available
- Cosmos Policy model loads successfully (2.0B params on CUDA) once TE + flash-attn are installed

---

## Tool Stack

| Layer | Tool | Status |
|-------|------|--------|
| **Cloud GPU** | RunPod A40 48GB | ✅ Active (Cosmos Policy) |
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
│   ├── openvla_oft.py      # OpenVLA-OFT adapter (continuous actions, Prismatic VLM)
│   ├── cosmos_policy.py    # Cosmos Policy adapter (video diffusion, CUDA-only)
│   └── pi0.py              # π0 adapter (KV cache patch, SigLIP shim)
├── vendor/
│   └── prismatic/          # Vendored Prismatic module for OFT
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
├── probe_results_openvla_widowx.json
├── probe_results_openvla_oft_franka.json
└── probe_results_cosmos_policy_franka.json

scripts/
├── setup_runpod.sh         # One-command RunPod A40 setup for CUDA models
requirements-runpod.txt     # Minimal deps for cloud GPU (no lerobot/streamlit)

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

### Phase 4.5: OpenVLA-OFT ✅
- Spawned coding agent to build adapter (`vla_probing/adapters/openvla_oft.py`, 391 lines)
- Vendored Prismatic module for VLM backbone compatibility
- All 8 probes complete on Franka scene
- **Key result:** Direction alignment 0.53 (vs base OpenVLA ~0.08 on WidowX)
- **Key insight:** VLM dead zone is backbone-level — Prismatic VLM insensitive to scene perturbations regardless of action head type
- Merged to main (commit `c374b8d`)

### Phase 4.7: Cosmos Policy ✅
- **Model:** `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` — video diffusion policy
- **Architecture:** Cosmos video encoder + T5-XXL text conditioning + DiT action head
- **Outputs:** 16-step action chunks + predicted future images + value estimates
- Adapter: `vla_probing/adapters/cosmos_policy.py` (branch `feat/cosmos-policy-adapter`)
- **Cloud GPU:** RunPod A40 48GB, pod `useless_aquamarine_moose` (200GB disk, 503GB RAM)
- **Working stack:** torch 2.6.0+cu124, flash-attn 2.7.4.post1 (prebuilt), TE 2.12.0 (from source)
- **All 8 probes complete** — results in `outputs/probes/probe_results_cosmos_policy_franka.json`
- **Key finding: Proprioceptive distribution fragility**
  - Standard Franka home qpos normalizes to [-4.4, -1.0, +2.6] under LIBERO stats
  - Model collapses to fixed trajectory — 0.0 perturbation sensitivity across all probes
  - π0 handles identical joint config with 0.47 direction alignment → fragility is Cosmos-specific
  - View ablation still shows partial visual processing (wrist camera ablation = 0.46 sensitivity)
  - Interpretation: real deployment concern, not test artifact — model requires exact LIBERO starting config
- **Bugs fixed during setup:**
  - Proprio dimension: Franka 8D → Cosmos expects 9D (padded with zero)
  - TE import crashes: patched load_framework_extension, DotProductAttention, distributed imports
  - Cosmos config: patched to skip base model download
  - EGL rendering: libegl1-mesa-dev + PyOpenGL for headless MuJoCo
  - torch version pinning: cosmos-policy pulls latest torch, must reinstall correct version after

### Phase 4.8: GR00T N1.6 (Next) 🔄
- **Model:** `nvidia/GR00T-N1.5-3B` — NVIDIA's foundation VLA
- **Architecture:** Cosmos-Reason-2B VLM + DiT action head
- **Adapter:** `vla_probing/adapters/groot.py` (commit `ffe9411`)
- **Plan:** Run on same RunPod pod, reuse existing CUDA stack
- **Status:** Adapter written, queued for execution

### Phase 5: Analysis & Content (Next)
- [x] Finish Cosmos Policy probes
- [ ] Run GR00T N1.6 probes on same RunPod instance
- [ ] Pull results back to Mac, merge branch to main
- [ ] Cross-model comparison analysis with 6 models
- [ ] Update dashboard for 6-model side-by-side view
- [ ] Blog post / writeup
- [ ] Share with Avik De
- [ ] Clean up 7 stale agent tmux sessions

### Phase 6: Future Models (Planned)
- [ ] GR00T N1.6 — run on same RunPod instance (Phase 4.8)
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

6. ✅ **Is the OpenVLA quantization dead zone from discretization or the VLM backbone?** — The VLM backbone. OpenVLA-OFT uses continuous actions with the same Prismatic VLM and still shows near-zero perturbation sensitivity (0.00008). The VLM itself doesn't propagate small visual differences into different representations.

7. ✅ **Do video-prediction models (Cosmos Policy) show better spatial understanding?** — No, at least not in zero-shot deployment scenarios. Cosmos Policy collapses to a fixed trajectory when proprioception is outside LIBERO's narrow training distribution. The video prediction backbone doesn't compensate for OOD proprioceptive input. Interestingly, view ablation shows the wrist camera still influences outputs (0.46 sensitivity), suggesting partial visual processing occurs even in the degenerate regime — but it's insufficient to produce meaningful actions.

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

### Confidence Levels

| Model | Confidence | Notes |
|-------|-----------|-------|
| **X-VLA** | High | Native embodiment, no workarounds |
| **π0** | Medium-high | Native Franka scene works well, some compatibility patches |
| **OpenVLA** | Medium | Temperature fix resolved determinism, quantization effects dominate |
| **OpenVLA-OFT** | Medium-high | Native Franka, continuous actions, vendored Prismatic module |
| **Cosmos Policy** | Pending | Model loads on CUDA, flash-attn building, probes imminent |
| **GR00T N1.6** | Pending | Adapter written, queued after Cosmos Policy on same RunPod |

---

*v6 — 2026-02-27 (updated RunPod setup to 200GB pod, Cosmos Policy loads on CUDA, T5-11b downloaded, flash-attn building, GR00T N1.6 adapter added, updated to 6-model comparison)*
