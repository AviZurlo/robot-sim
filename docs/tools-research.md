# VLA Diagnostic Probes — Tools & Frameworks Research

*Last updated: 2026-02-26*

## Executive Summary

We're probing 4 VLA models (X-VLA, π0, SmolVLA, OpenVLA) to understand what they learn. This doc covers existing tools we can leverage to avoid building from scratch. **Key finding: LeRobot is the unifying framework** — it natively supports 3 of our 4 models (X-VLA, π0, SmolVLA) and has built-in evaluation infrastructure. OpenVLA uses HF Transformers directly.

---

## 1. VLA Evaluation Frameworks

### LeRobot (★ Primary Framework)
- **Built-in eval:** `lerobot-eval` command supports simulation evaluation out of the box
- **Supported envs:** LIBERO, MetaWorld, and custom envs via EnvHub (Gymnasium-compatible)
- **VLA support:** Pi0/Pi0Fast, SmolVLA, X-VLA all have native LeRobot implementations
- **EnvHub:** New feature — load simulation envs from HF Hub with one line of code. Custom envs can be published and shared. Uses standard `gym.vector.VectorEnv` interface.
- **No built-in probing/interpretability tools** — we'll need to build the probe layer ourselves

### SimplerEnv (CoRL 2024)
- **What it does:** Real-to-sim evaluation using SAPIEN/ManiSkill2. Visual matching + variant aggregation setups.
- **Supported robots:** Google Robot, WidowX (Bridge)
- **Supported models (out of box):** RT-1, RT-1-X, Octo
- **Our models:** NOT directly supported, but X-VLA has a `2toINF/X-VLA-Google-Robot` checkpoint and `lerobot/xvla-widowx`. OpenVLA works on WidowX/Bridge. Adding new policies is documented.
- **GPU parallel version:** ManiSkill3 branch runs 10-15x faster
- **Verdict:** Useful for WidowX-based evaluation; requires writing policy wrappers for our models

### RoboVLMs (arXiv 2412.14058)
- **What it is:** Systematic VLA comparison framework — 8 VLM backbones, 4 policy architectures, 600+ experiments
- **Benchmarks used:** CALVIN, SimplerEnv, real-world
- **Open-source:** Code, models, datasets, toolkits at robovlms.github.io
- **Verdict:** Most relevant existing comparison work. Their framework could accelerate our probe design. Worth examining their evaluation code.

### Other Benchmarks
- **LIBERO:** Supported by LeRobot eval. 4 task suites (spatial, object, goal, long). Good for testing spatial understanding.
- **CALVIN:** Long-horizon benchmark. X-VLA has a CALVIN checkpoint (`2toINF/X-VLA-Calvin-ABC_D`).
- **RLBench:** More complex, 100+ tasks, but heavier setup. Less relevant for our focused probes.

### ⭐ Recommendation
Use **LeRobot** as the base framework with **LIBERO** as the primary evaluation environment (all models can be evaluated there). Use SimplerEnv for WidowX-specific probes if needed.

---

## 2. Attention Visualization Tools

### For HuggingFace Transformer Models

| Tool | Vision-Language Support | Effort to Adapt | Notes |
|------|----------------------|-----------------|-------|
| **Captum** (Facebook) | ✅ Model-agnostic | Low | Best overall. Supports integrated gradients, attention attribution, GradCAM. Works with any PyTorch model. |
| **BertViz** | ⚠️ Text-focused | Medium | Great attention head visualization but designed for NLP. Can adapt for VLMs by treating image tokens as sequence elements. |
| **Attention Rollout** | ✅ Works with ViTs | Low | Simple technique — multiply attention matrices across layers. Good for understanding where vision tokens attend. |
| **GradCAM / ScoreCAM** | ✅ Vision models | Low | Standard for CNN/ViT visual explanations. `pytorch-grad-cam` library supports ViTs. |
| **transformer-interpret** | ⚠️ Limited | High | Mostly for classification tasks, not action prediction. |

### VLA-Specific Considerations
- **X-VLA:** Uses Florence 2 encoder + Soft-Prompted Transformer. Attention in the denoiser is where embodiment-specific behavior lives. Extract attention from `SoftPromptedTransformer` layers.
- **π0:** Uses PaliGemma VLM backbone + flow matching action head. Attention from PaliGemma vision encoder is accessible via standard HF attention outputs.
- **SmolVLA:** Built on SmolVLM (small VLM). Same HF attention extraction approach.
- **OpenVLA:** Uses Prismatic VLM (DINOv2 + SigLIP) → Llama-2. Standard HF `output_attentions=True` should work.

### ⭐ Recommendation
Use **Captum** for attribution analysis (model-agnostic, works with all 4 models). Use **custom attention extraction** via `output_attentions=True` in HF models + matplotlib/plotly for visualization. Write a thin wrapper per model (~50 lines each) to extract attention maps from the right layers.

---

## 3. Experiment Tracking

| Tool | Image/Traj Artifacts | Cross-Model Compare | Setup Effort | Notes |
|------|---------------------|---------------------|-------------|-------|
| **Weights & Biases** | ✅ Excellent | ✅ Best-in-class | Low | Tables, images, 3D plots, custom panels. Side-by-side comparison is a core feature. Free for academic. |
| **Aim** | ✅ Good | ✅ Good | Low | Open-source, self-hosted. Great image tracking. Lighter than W&B. |
| **TensorBoard** | ⚠️ Basic | ⚠️ Clunky | Minimal | Already in most ML stacks. Image support exists but comparison UI is weak. |
| **MLflow** | ✅ Good | ⚠️ OK | Medium | Better for production ML than research exploration. |
| **Neptune** | ✅ Good | ✅ Good | Low | Similar to W&B but less community adoption. |

### ⭐ Recommendation
**Weights & Biases** — best for our use case. Key features:
- Log trajectory images, attention heatmaps, action distributions as artifacts
- W&B Tables for structured probe results (model × probe × metric)
- Side-by-side image comparison panels
- Already supported by LeRobot (just set `wandb` in training config)

If you want fully self-hosted/offline: **Aim** is the best open-source alternative.

---

## 4. Trajectory Analysis Tools

### Metrics Libraries
- **scipy.spatial.distance** — DTW via `scipy.spatial.distance.cdist` + `dtw-python` package
- **tslearn** — DTW, trajectory clustering, time series metrics
- **dtw-python** — Dedicated DTW library with various step patterns
- **numpy** — L2 error, smoothness (finite differences for jerk = d³x/dt³)

### Visualization
- **matplotlib** — 3D trajectory plots (`mpl_toolkits.mplot3d`), overlaid comparisons
- **plotly** — Interactive 3D trajectories, great for comparing predicted vs ground truth
- **rerun.io** — 3D robotics visualization, supports trajectories + point clouds + images in sync. Modern alternative to RViz.

### Robotics-Specific
- **robosuite** — Has built-in trajectory recording and playback
- **ManiSkill** — Trajectory recording in SimplerEnv evaluations
- **LeRobot datasets** — Each episode stores actions + observations, easy to extract trajectories

### Smoothness Metrics (implement yourself, ~20 lines)
```python
def trajectory_jerk(traj, dt=1.0):
    """Compute mean absolute jerk (3rd derivative) as smoothness metric."""
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return np.mean(np.abs(jerk))
```

### ⭐ Recommendation
- **dtw-python** for DTW metric
- **plotly** for interactive trajectory visualization
- **numpy** for L2 error + custom smoothness/jerk metrics
- Consider **rerun.io** if you want synchronized video + trajectory replay

---

## 5. Existing VLA Comparison Work

### Most Relevant Papers/Repos

1. **RoboVLMs** (arXiv 2412.14058) — Most comprehensive VLA comparison to date. 8 VLM backbones, 4 architectures, CALVIN + SimplerEnv + real. Open-source framework.
   - Repo: https://robovlms.github.io
   - Includes evaluation code for multiple VLA architectures

2. **X-VLA paper** (arXiv 2510.10274) — Compares X-VLA against OpenVLA, Octo, and others on SimplerEnv + CALVIN + LIBERO. Has evaluation code.

3. **SimplerEnv paper** (CoRL 2024) — Establishes real-to-sim correlation for policy evaluation. Compares RT-1, RT-1-X, Octo.

4. **Open X-Embodiment** — The dataset collection. Provides cross-embodiment benchmarks but not VLA-specific probing.

5. **SmolVLA paper** (arXiv 2506.01844) — Compares against larger VLAs on community data.

### No Existing "VLA Zoo" or Unified Comparison Notebook
Nobody has done exactly what we're doing — probing *what* VLAs learn rather than just measuring task success. This is novel.

---

## 6. Each Model's Native Embodiment

### X-VLA (0.9B params, flow-matching)
- **Base checkpoint:** `2toINF/X-VLA-Pt` (pretrained on heterogeneous data)
- **LeRobot checkpoints:** `lerobot/xvla-base`, `lerobot/xvla-widowx`, `lerobot/xvla-google-robot`, `lerobot/xvla-agibot-world`, `lerobot/xvla-folding`
- **Available embodiment checkpoints:**
  - WidowX (Bridge) — `2toINF/X-VLA-WidowX` / `lerobot/xvla-widowx`
  - Google Robot — `2toINF/X-VLA-Google-Robot` / `lerobot/xvla-google-robot`
  - LIBERO (Franka Panda) — `2toINF/X-VLA-Libero`
  - CALVIN (Franka Panda) — `2toINF/X-VLA-Calvin-ABC_D`
  - RoboTwin2, VLABench, SoftFold, AgiBot World
- **Training data:** Bridge Data (primary), with soft-prompt adaptation for each embodiment
- **Best for zero-shot:** WidowX (primary training embodiment)

### π0 (lerobot/pi0_base)
- **Architecture:** PaliGemma VLM backbone + flow matching action head
- **Base checkpoint:** `lerobot/pi0_base` — intended as a base model for fine-tuning
- **Available fine-tuned:** `lerobot/pi0_libero_finetuned`
- **Training data:** Not fully disclosed (Physical Intelligence proprietary data), but the open-source version is meant for fine-tuning on your own data
- **Supported in LeRobot:** LIBERO evaluation confirmed working
- **Best for evaluation:** LIBERO (has fine-tuned checkpoint). For zero-shot: limited — pi0_base needs fine-tuning

### SmolVLA (~small, efficient)
- **Architecture:** SmolVLM (small VLM) + flow matching action head
- **Base checkpoint:** `lerobot/smolvla_base`
- **Training data:** Community-collected data from affordable robotic platforms (SO-100 and similar LeRobot community robots)
- **Design goal:** Single-GPU trainable, deployable on consumer hardware/CPUs
- **Key limitation:** Trained primarily on community data (SO-100, LeKiwi) — NOT on standard benchmarks like Bridge or LIBERO
- **Best for evaluation:** Needs fine-tuning for any standard benchmark. This is the most challenging model for fair comparison.

### OpenVLA-7B
- **Architecture:** Prismatic VLM (DINOv2 + SigLIP) → Llama-2 → discretized actions
- **Base checkpoint:** `openvla/openvla-7b`
- **Training data:** 970K episodes from Open X-Embodiment (many embodiments)
- **Zero-shot embodiments:** WidowX/BridgeV2 (primary, best supported), Google Robot, and others from OXE mix
- **Action format:** 7-DoF end-effector deltas (x, y, z, roll, pitch, yaw, gripper)
- **Key note:** Does NOT zero-shot generalize to unseen embodiments
- **Best for zero-shot:** WidowX / BridgeV2 (unnorm_key="bridge_orig")

### ⭐ Recommendation: Best Embodiment for Fair Comparison

**WidowX / BridgeV2 is the only viable common ground for zero-shot:**
- X-VLA: ✅ Primary training embodiment, dedicated checkpoint
- π0: ⚠️ Base model can be fine-tuned on Bridge; no zero-shot Bridge checkpoint available
- SmolVLA: ❌ Not trained on Bridge data
- OpenVLA: ✅ Primary zero-shot embodiment

**LIBERO (Franka Panda) for fine-tuned comparison:**
- X-VLA: ✅ Has LIBERO checkpoint
- π0: ✅ Has LIBERO fine-tuned checkpoint
- SmolVLA: ⚠️ Would need fine-tuning
- OpenVLA: ⚠️ Would need fine-tuning

**Practical recommendation:** Use **LIBERO** as the primary benchmark:
1. X-VLA and π0 have existing LIBERO checkpoints
2. OpenVLA and SmolVLA can be fine-tuned on LIBERO with minimal data
3. LIBERO has spatial/object/goal/long-horizon task suites — perfect for our probes
4. LIBERO runs in MuJoCo, easy to set up

For pure zero-shot probing: Use **WidowX/Bridge** with X-VLA and OpenVLA only (2-model comparison), noting π0 and SmolVLA aren't fairly comparable without fine-tuning.

---

## 7. MuJoCo Simulation Environments

### By Robot Platform

| Robot | Sim Environment | Framework | Our Models |
|-------|----------------|-----------|------------|
| **Franka Panda (LIBERO)** | LIBERO benchmark | MuJoCo + robosuite | X-VLA ✅, π0 ✅, OpenVLA (finetune), SmolVLA (finetune) |
| **WidowX (Bridge)** | SimplerEnv | SAPIEN/ManiSkill2 | X-VLA ✅, OpenVLA ✅ |
| **Google Robot** | SimplerEnv | SAPIEN/ManiSkill2 | X-VLA ✅ |
| **ALOHA** | gym-aloha | MuJoCo/Gymnasium | π0 (community) |
| **SO-100** | No standard sim yet | — | SmolVLA (native robot) |

### LeRobot EnvHub
LeRobot's new EnvHub system lets you load simulation environments directly from HF Hub:
```python
from lerobot.envs.factory import make_env
env = make_env("lerobot/cartpole-env", trust_remote_code=True)
```
Check for LIBERO env on Hub: `lerobot-eval --env.type=libero` is supported natively.

### Available Packages
- **gym-aloha** — ALOHA bimanual tasks in MuJoCo
- **gym-xarm** — xArm tasks (used in some LeRobot examples)
- **LIBERO** — `pip install libero`, 4 task suites, MuJoCo-based
- **SimplerEnv** — Google Robot + WidowX in SAPIEN, real-to-sim visual matching
- **CALVIN** — Franka Panda, long-horizon, MuJoCo-based

### ⭐ Recommendation
**LIBERO** is the path of least resistance:
1. MuJoCo-based, easy to install (`pip install libero`)
2. Native LeRobot support via `lerobot-eval --env.type=libero`
3. 4 task suites align perfectly with our probes:
   - `libero_spatial` → spatial understanding probes
   - `libero_object` → object recognition probes
   - `libero_goal` → goal-directed behavior
   - `libero_long` → long-horizon planning
4. Two models already have LIBERO checkpoints

For WidowX zero-shot evaluation, use **SimplerEnv** with the ManiSkill3 branch for GPU acceleration.

---

## Summary: Recommended Tool Stack

| Layer | Tool | Why |
|-------|------|-----|
| **Framework** | LeRobot | Unifies 3/4 models, has eval infrastructure |
| **Simulation** | LIBERO (primary), SimplerEnv (WidowX) | Best model coverage, aligned task suites |
| **Attention/Attribution** | Captum + custom extraction | Model-agnostic, works with all 4 |
| **Trajectory Analysis** | dtw-python + plotly + numpy | Lightweight, sufficient |
| **Experiment Tracking** | Weights & Biases | Best comparison UI, image/artifact support |
| **VLA Comparison Reference** | RoboVLMs framework | Most comprehensive existing comparison |

### What We Need to Build (Not Available Off-the-Shelf)
1. **Probe harness:** Unified interface to run each probe across all 4 models
2. **Attention extraction wrappers:** Per-model code to extract attention maps from the right layers
3. **Perturbation generators:** Camera/prompt perturbation pipeline
4. **Metrics aggregation:** Collect probe results into W&B tables for comparison
5. **Model loading unification:** Thin wrapper to load all 4 models with consistent predict API

These are all lightweight (~200-500 lines total) given the existing tooling.
