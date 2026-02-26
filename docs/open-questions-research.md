# VLA Models on Mac Mini M4 (24GB) — Open Questions Research

**Date:** 2026-02-26  
**Context:** Running diagnostic probes across VLA models, inspired by [Debugging as Architecture Insight](https://www.avikde.me/p/debugging-as-architecture-insight)  
**Hardware:** Mac mini M4, 24GB unified memory, PyTorch 2.7.1, MPS backend

---

## Question 1: MPS Compatibility for X-VLA

### Findings

**X-VLA architecture:** 0.9B params, uses InternVL2 as VLM backbone + flow matching action head + BART tokenizer for language.

**LeRobot implementation:** The sample code in `lerobot/xvla-widowx` defaults to `cuda` with CPU fallback — no explicit MPS path. The code does `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

**Known MPS risks:**
- **InternVL2** uses dynamic resolution and complex attention patterns. MPS historically has issues with certain attention implementations (e.g., `torch.nn.functional.scaled_dot_product_attention` with custom masks can fail on MPS).
- **Flow matching action head** uses continuous diffusion-style denoising. Basic operations (linear layers, activations) should be fine. Risk is in any custom CUDA kernels or ops like `torch.linalg` calls that may not have MPS implementations.
- **flash_attention_2** (used by InternVL2) does NOT work on MPS — would need to fall back to `eager` or `sdpa` attention.
- No MPS-specific issues found in GitHub issues for `2toinf/X-VLA` or lerobot (but also no MPS success reports).

**Avik's repo** (`avikde/vla-pipeline`) uses CUDA 12.8 with an RTX 5070 Ti — no MPS testing evident.

### Recommendation
- **Likely feasible with patches.** X-VLA is only 0.9B params (~1.8GB fp16) — fits easily in 24GB.
- Force `attn_implementation="eager"` when loading InternVL2 to avoid flash_attn issues.
- Replace `device = "cuda"` with `"mps"` and test. Expect 1-3 ops to fail; most can be worked around with CPU fallbacks for specific operations.
- **This is the best candidate to start with** due to small size.

---

## Question 2: OpenVLA-7B Memory on 24GB Mac

### Findings

**Model size:** 7B params (Llama-2 backbone + DINOv2 + SigLIP vision encoders). fp16 weights ≈ 14GB.

**Peak memory during inference:** With attention KV cache for a single forward pass, expect ~16-18GB. Autoregressive action token generation (7 tokens for 7-DoF) adds modest overhead.

**bitsandbytes on MPS:** **Does not work.** bitsandbytes requires CUDA. This has been the case historically and remains so as of early 2026. No MPS backend support.

**Alternative quantization for MPS:**
- **MLX:** Best option for Apple Silicon. The community has quantized many Llama-2 based models to 4-bit via MLX. However, OpenVLA's custom architecture (VLM + action head) would need porting to MLX — non-trivial.
- **GGUF/llama.cpp:** Only works for the LLM backbone, not the full VLA pipeline with vision encoders.
- **CoreML:** Possible via `coremltools` but requires full model conversion. No existing OpenVLA CoreML model.
- **PyTorch native quantization:** `torch.ao.quantization` supports some int8 ops on MPS as of PyTorch 2.x, but coverage is incomplete.

**Smaller variants:**
- **OpenVLA-OFT** (March 2025): Optimized Fine-Tuning recipe. Still 7B params but 25-50x faster inference via continuous actions (no autoregressive decoding). Same memory footprint but much faster.
- No smaller (e.g., 3B) official OpenVLA variant exists.

### Recommendation
- **Tight but possible at fp16** (~16-18GB of 24GB). Leave little room for MuJoCo rendering.
- **fp32 will NOT fit.** Must use fp16 or bf16 (MPS supports both in PyTorch 2.7).
- Best approach: load with `torch_dtype=torch.float16` and hope peak stays under ~20GB.
- **Deprioritize OpenVLA for initial experiments.** X-VLA (0.9B) and π0 (3B) are better fits for 24GB.

---

## Question 3: Scene Setup — Static Images vs MuJoCo Rendering

### Findings

**Avik's repo (`avikde/vla-pipeline`):**
- **Uses MuJoCo rendering**, not static images. The `demo_xvla_widowx.py` script loads a WidowX MuJoCo XML model (`assets/widowx/widowx_vision_scene.xml`), runs physics simulation, and renders from cameras at each timestep.
- Renders at **256×256** resolution for VLA input.
- Uses two camera views: `"up"` and `"side"` (matching X-VLA WidowX training on BridgeData).
- The Jupyter notebook likely uses MuJoCo + interactive widgets for visualization.

**MuJoCo for WidowX:**
- Avik already has a WidowX MJCF scene file with cameras and objects — this is directly reusable.
- MuJoCo runs natively on macOS/ARM64 with excellent performance.
- `dm_control` and `mujoco` Python packages both work on Apple Silicon.

**Image format/resolution expected by VLAs:**
| Model | Resolution | Format |
|-------|-----------|--------|
| X-VLA | 256×256 | RGB tensor, float [0,1] |
| OpenVLA | 224×224 | RGB, normalized per Prismatic VLM |
| π0 | Likely 224×224 or 256×256 | RGB tensor |
| SmolVLA | 224×224 | RGB tensor |

### Recommendation
- **Fork/clone `avikde/vla-pipeline`** — it already has the MuJoCo WidowX scene, rendering pipeline, and X-VLA integration.
- Adapt for MPS (change device selection) and add our diagnostic probes.
- MuJoCo rendering on Mac is fast (~1ms per frame at 256×256) and gives full control over scene parameters.

---

## Question 4: π0 Weights Availability

### Findings

**Available through LeRobot:**
- **`lerobot/pi0_base`** — Base π0 model, Apache 2.0 license. Available on HuggingFace.
- **`lerobot/pi0_libero`** — Fine-tuned on LIBERO dataset.
- Install: `pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"`

**Architecture:** 3B parameter VLM backbone + flow matching action expert. Continuous actions.

**Supported embodiments (training data):** UR5e, Bimanual UR5e, Franka, Bimanual Trossen, Bimanual ARX, Mobile Trossen, Mobile Fibocom. **No WidowX-specific checkpoint** found.

**WidowX compatibility:** π0 is cross-embodiment and uses Open X-Embodiment data (which includes BridgeData/WidowX). The base model may handle WidowX zero-shot, but no dedicated fine-tuned checkpoint exists.

**Memory requirements:**
- 3B params at fp16 ≈ **6GB** for weights.
- With KV cache and flow matching denoising: estimate **8-12GB** peak during inference.
- **Fits comfortably in 24GB on MPS.**

**Reference implementation:** [OpenPI](https://github.com/Physical-Intelligence/openpi) — original Physical Intelligence code.

### Recommendation
- **π0 is the second-best candidate** after X-VLA for 24GB Mac.
- Use `lerobot/pi0_base` and test with WidowX scenes. Even without a WidowX-specific checkpoint, the cross-embodiment training should produce reasonable actions.
- Same MPS caveats apply (flash_attn → eager, potential op gaps).

---

## Question 5: Quantitative Evaluation Metrics

### Findings

**Standard VLA evaluation metrics in the literature:**
1. **Task success rate** — binary: did the robot complete the task? (gold standard, requires simulation loop)
2. **L2 action error** — Euclidean distance between predicted and ground-truth actions (from dataset episodes)
3. **Trajectory-level metrics:**
   - Mean L2 displacement error over action chunks
   - Dynamic Time Warping (DTW) distance to reference trajectories
4. **For stochastic models (π0, X-VLA with flow matching):**
   - Action variance across N forward passes with different noise seeds
   - Mode coverage (do different seeds produce diverse but valid trajectories?)
5. **Gripper accuracy** — binary classification accuracy for open/close
6. **Smoothness** — jerk (3rd derivative of position) of predicted trajectories

**What OpenVLA uses:** Success rate on LIBERO benchmark + BridgeData V2 real-world evaluations. Their paper also reports per-task success rates.

**What Avik's repo has:** From the code, primarily **visualization** — rendering predicted trajectories overlaid on scenes. The notebook appears to be qualitative (visual inspection with sliders). No quantitative metrics implementation found in the scripts.

**For diagnostic probes specifically (per the blog post concept):**
- Attention map analysis (which image regions does the model attend to?)
- Sensitivity to instruction perturbation (change one word → how much does action change?)
- Proprioception ablation (zero out proprio → measure action delta)
- These are more about interpretability than task performance.

### Recommendation

**Implement these metrics in order of usefulness:**

1. **L2 action error vs dataset ground truth** — easiest, most informative baseline. Load BridgeData episodes, run inference, compare.
2. **Stochastic trajectory spread** — for flow matching models (X-VLA, π0), run 10 seeds and plot action variance. This directly probes the model's uncertainty.
3. **Attention visualization** — extract attention maps to understand what the model "sees."
4. **Perturbation sensitivity** — systematic ablations (change instruction, mask image regions, zero proprio) to probe architecture behavior.
5. **Task success rate** — requires a full simulation loop with reward detection. More complex but ultimate measure.

---

## Summary: Recommended Model Priority

| Model | Params | fp16 Size | Fits 24GB? | MPS Risk | Priority |
|-------|--------|-----------|------------|----------|----------|
| **X-VLA** | 0.9B | ~1.8GB | ✅ Easily | Medium (InternVL2 attention) | **#1 — Start here** |
| **π0** | 3B | ~6GB | ✅ Yes | Medium (similar flow matching) | **#2** |
| **SmolVLA** | ~0.5B | ~1GB | ✅ Easily | Low (simpler arch) | **#3 (if available for WidowX)** |
| **OpenVLA-7B** | 7B | ~14GB | ⚠️ Tight | High (no quantization on MPS) | **#4 — Deprioritize** |

**Starting point:** Clone `avikde/vla-pipeline`, adapt for MPS, run X-VLA first. The MuJoCo scene and rendering pipeline are already there.
