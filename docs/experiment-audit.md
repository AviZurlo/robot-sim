# VLA Probing Experiment Methodology Audit

**Date**: February 26, 2025  
**Auditor**: Claude (Anthropic)  
**Project**: robot-sim VLA probing suite  
**Models Evaluated**: X-VLA, π0, SmolVLA, OpenVLA  

## Executive Summary

This audit evaluates the experimental methodology of 8 diagnostic probes across 4 Vision-Language-Action (VLA) models. The experiments test model behaviors including baseline performance, spatial understanding, camera sensitivity, attention patterns, and robustness to perturbations.

**Key Findings:**
- ✅ **Methodologically sound** overall experimental design
- ⚠️  **Critical issues** with OpenVLA setup (deterministic behavior, identical outputs)
- ⚠️  **Complex compatibility shims** required for π0 (transformers version conflicts)
- ⚠️  **Cross-embodiment** domain gaps for SmolVLA (trained on different robots)
- ✅ **Robust evaluation** framework with standardized interfaces and metrics

## 1. Adapter Setup Analysis

### 1.1 X-VLA Adapter (`lerobot/xvla-widowx`)

**✅ Properly Configured**

- **Checkpoint**: `lerobot/xvla-widowx` (HuggingFace/lerobot)
- **Device**: MPS with CUDA/CPU fallback via `_get_device()`
- **Precision**: Auto-detected (likely fp16 on MPS)
- **Architecture**: Florence2 + DaViT vision encoder + flow matching
- **Input preprocessing**:
  - Images: 2 camera views, (H,W,3) uint8 → (1,3,H,W) float32 [0,1]
  - State: 8D BridgeData format [x,y,z,roll,pitch,yaw,0,gripper]
  - Language: Tokenized with max_length from config
- **Action space**: 20D (2 timesteps × 10D each), chunk_size=30
- **Workarounds**: None required

### 1.2 π0 Adapter (`lerobot/pi0_libero`)

**⚠️ Requires Complex Compatibility Shims**

- **Checkpoint**: `lerobot/pi0_libero` (LIBERO fine-tuned, 3B params)
- **Device**: MPS with fallbacks
- **Precision**: Auto-detected
- **Architecture**: PaliGemma 3B + Gemma-300M action expert + flow matching
- **Input preprocessing**:
  - Images: 2 camera views (+ 1 empty padded), same format as X-VLA
  - State: 8D → padded to 32D internally
  - Language: Gemma tokenizer with newline suffix requirement
- **Action space**: 7D (EE pos + rotation + gripper), chunk_size=50
- **Critical workarounds**:
  1. **SigLIP check shim** - Creates fake `transformers.models.siglip.check` module
  2. **Denoise step patching** - Deep-copies KV cache to prevent growth across denoising steps
  3. **Tokenizer fallback** - Uses `unsloth/gemma-2b` when PaliGemma is inaccessible

**🚨 Red Flag**: The need for these extensive patches indicates fragile dependencies and potential instability.

### 1.3 SmolVLA Adapter (`lerobot/smolvla_base`)

**⚠️ Cross-Embodiment Domain Gap**

- **Checkpoint**: `lerobot/smolvla_base` (~0.5B params)
- **Device**: MPS with fallbacks  
- **Precision**: Auto-detected
- **Architecture**: SmolVLM2-500M + cross-attention action expert + flow matching
- **Input preprocessing**:
  - Images: Maps to config-defined camera keys, pads missing cameras with first image
  - State: 8D → padded to 32D
  - Language: SmolVLM2 tokenizer with newline suffix
- **Action space**: Variable (config-dependent, typically 6D for SO-100)
- **Workarounds**: None in adapter, but monkey-patches attention capture

**⚠️ Critical Issue**: SmolVLA was trained on SO-100 community data (non-WidowX), making this **cross-embodiment** evaluation. Results should be interpreted as domain transfer, not task performance.

### 1.4 OpenVLA Adapter (`openvla/openvla-7b`)

**🚨 Critical Methodological Issues**

- **Checkpoint**: `openvla/openvla-7b` (7B params)
- **Device**: MPS with memory checking, fallback to CPU
- **Precision**: fp16 (MPS) or fp32 (CPU)
- **Architecture**: Prismatic (DINOv2 + SigLIP) + Llama-2 7B
- **Input preprocessing**:
  - Images: Single camera only (different from other models!)
  - State: **Not used** (OpenVLA doesn't take proprioception)
  - Language: OpenVLA-specific prompt template
- **Action space**: 7D, chunk_size=1 (autoregressive, not chunked)
- **Workarounds**: None required

**🚨 Red Flags**: 
1. **Deterministic output**: All metrics show 0.0 for sensitivity/variance measures
2. **Identical trajectories**: Same output regardless of prompt/scene changes
3. **Single camera**: Uses different input modality than other models

## 2. Scene Configuration Analysis

### 2.1 MuJoCo WidowX Scene

**✅ Well-Designed Test Environment**

- **XML File**: `widowx_vision_scene.xml`
- **Embodiment**: WidowX arm simulation
- **Cameras**: 
  - Primary: "up" (over-shoulder view, BridgeData-compatible)
  - Secondary: "side" (angled view)
- **Resolution**: 256×256 (standard VLA input size)
- **Objects**: Red and blue blocks with manipulatable positions
- **State**: 8D BridgeData format [x,y,z,roll,pitch,yaw,0,gripper]

### 2.2 Embodiment Compatibility Analysis

| Model | Training Embodiment | Scene Embodiment | Image Keys | State Dims | Action Dims | Compatible? |
|-------|-------------------|------------------|------------|------------|-------------|-------------|
| X-VLA | WidowX (BridgeData) | WidowX (MuJoCo) | ✅ image, image2 | ✅ 8D→8D | ✅ 20D | **Perfect** |
| π0 | Franka Panda (LIBERO) | WidowX (MuJoCo) | ✅ image, image2 | ✅ 8D→8D | ⚠️ 7D vs training | **Structural** |
| SmolVLA | Various (SO-100) | WidowX (MuJoCo) | ⚠️ Flexible mapping | ✅ Variable→32D | ⚠️ Variable | **Cross-embodiment** |
| OpenVLA | Mixed datasets | WidowX (MuJoCo) | ⚠️ Single camera only | ❌ Not used | ✅ 7D | **Partial** |

**Key Issues:**
- **π0**: Expects Franka Panda visuals but gets WidowX MuJoCo (domain gap)
- **SmolVLA**: Trained on community robots, not WidowX (major domain gap)
- **OpenVLA**: Uses only 1 camera vs 2 for others, ignores proprioception

## 3. Probe Execution Analysis

### 3.1 Probe Implementation Review

**✅ Standardized Framework**
- All probes inherit from `Probe` base class
- Consistent `VLAInput`/`VLAOutput` interface
- Standardized metrics computation via `compute_all_metrics()`

### 3.2 Individual Probe Analysis

#### Probe 1: Baseline (`pick_up_red_block`)
- **✅ Correctly implemented**: Single prediction + multi-seed analysis
- **✅ Proper trajectory extraction**: Handles chunked vs single-step outputs
- **✅ Target-oriented metrics**: Direction alignment, distance to target

#### Probe 2: Spatial Symmetry (swap block positions)
- **✅ Valid methodology**: Compares baseline vs swapped trajectories
- **✅ DTW distance**: Appropriate trajectory comparison metric
- **✅ Scene manipulation**: Proper block position swapping

#### Probe 3: Camera Sensitivity (mirror camera/flip image)
- **✅ Dual perturbations**: Tests both camera geometry and image processing
- **✅ Sensitivity metrics**: Measures trajectory changes under visual perturbations

#### Probe 4: View Ablation (zero out camera views)
- **✅ Systematic ablation**: Primary, secondary, and full vision ablation
- **✅ Multi-modal analysis**: Tests vision dependency properly

#### Probe 5: Counterfactual (synonym prompts)
- **✅ Language robustness**: Tests sensitivity to prompt variations
- **✅ Comprehensive synonyms**: "red cube", "crimson block", "grasp", etc.

#### Probe 6: Null Action (non-movement prompts)
- **✅ Baseline comparison**: Tests null vs active prompt responses
- **✅ Diverse null prompts**: "don't move", "stay still", "do nothing", etc.

#### Probe 7: Attention (spatial attention analysis)  
- **✅ Cross-model extraction**: Different attention mechanisms per architecture
- **⚠️ Complex hooks**: Model-specific monkey-patching required
- **❌ Failed for π0/SmolVLA**: Returns all-zero attention (fallback used)

#### Probe 8: Perturbation (block position shifts)
- **✅ Systematic perturbations**: Left/right/forward/back/up displacements
- **✅ Correlation analysis**: Sensitivity vs displacement correlation

### 3.3 Trajectory Generation

**✅ Correct Implementation**
- **Multi-step rollout**: All models predict action chunks, extract XYZ trajectories
- **Proper seed handling**: Multi-seed analysis for stochastic models (X-VLA, π0, SmolVLA)
- **Error handling**: Graceful fallbacks for attention extraction failures

## 4. Results Validation Analysis

### 4.1 Cross-Model Result Comparison

| Metric | X-VLA | π0 | SmolVLA | OpenVLA | Expected Range | Assessment |
|--------|-------|----|---------|---------|--------------.|------------|
| **Trajectory Jerk** | 0.0003 | 1.507 | 0.056 | 0.0 | [0, 2] | ⚠️ OpenVLA suspicious |
| **Distance to Target** | 0.058 | 0.992 | 1.380 | 0.270 | [0, 1.5] | ✅ Reasonable |
| **Direction Alignment** | 0.142 | -0.015 | -0.957 | N/A | [-1, 1] | ✅ Varied |
| **Trajectory Spread** | 0.0004 | 0.443 | 0.179 | 5.78e-19 | [0, 0.5] | 🚨 OpenVLA zero |

### 4.2 Stochasticity Analysis

**X-VLA**: ✅ Low but non-zero variance (flow matching working correctly)
**π0**: ✅ High variance (flow matching very stochastic) 
**SmolVLA**: ✅ Moderate variance (appropriate stochasticity)
**OpenVLA**: 🚨 **Zero variance across all probes** - completely deterministic

### 4.3 Sensitivity Metrics

**Perturbation Sensitivity:**
- X-VLA: 0.00156 (low, stable)
- π0: 0.00639 (moderate) 
- SmolVLA: 0.103 (high, unstable)
- OpenVLA: 0.0 (🚨 **no response to perturbations**)

### 4.4 Attention Analysis

**Successful Extraction:**
- ✅ X-VLA: IoU = 0.173 (reasonable spatial attention)
- ❌ π0: IoU = 0.0 (fallback used, attention hooks failed)
- ❌ SmolVLA: IoU = 0.0 (fallback used)  
- ✅ OpenVLA: IoU = 0.109 (some spatial attention detected)

## 5. Red Flags and Critical Issues

### 5.1 🚨 OpenVLA Critical Issues

**Deterministic Behavior**: OpenVLA shows **identical outputs** across all conditions:
- Zero trajectory variance across seeds
- Zero sensitivity to spatial perturbations  
- Zero sensitivity to camera changes
- Zero sensitivity to prompt variations
- **All actions identical: [same 7D vector]**

**Possible Causes:**
1. **Model frozen/cached**: Actions generated from cached/static weights
2. **Autoregressive sampling**: `do_sample=False` forcing deterministic decoding
3. **Discretization artifacts**: Action space quantization causing identical bins
4. **Implementation bug**: Missing stochasticity in text-to-action decoding

**Recommendation**: 🚨 **OpenVLA results are invalid** - investigate deterministic behavior before trusting any metrics.

### 5.2 ⚠️ π0 Dependency Issues  

**Fragile Setup**: Requires extensive monkey-patching:
- Custom transformers fork compatibility shim
- KV cache deep-copying to prevent memory growth
- Tokenizer fallbacks for gated models

**Risk**: These patches may mask underlying issues or introduce subtle bugs.

### 5.3 ⚠️ Cross-Embodiment Domain Gaps

**SmolVLA**: Trained on SO-100 (community robots) but tested on WidowX
**π0**: Trained on LIBERO (Franka Panda) but tested on WidowX

**Impact**: Results reflect **domain transfer performance**, not native task capability.

### 5.4 ⚠️ Attention Extraction Failures

**π0 & SmolVLA**: Attention hooks failed → all-zero IoU metrics
**Cause**: Complex model architectures require model-specific extraction methods
**Impact**: Attention-based probes invalid for these models

### 5.5 ⚠️ Scene Validation Gaps

**Missing Checks:**
- No validation that block positions match expected locations
- No verification that camera views render correctly  
- No sanity checks on action space bounds

## 6. Methodological Strengths

### 6.1 ✅ Robust Evaluation Framework
- Standardized adapter interface enables fair comparison
- Comprehensive metrics suite (jerk, DTW, IoU, sensitivity)
- Multi-seed analysis captures stochasticity properly
- Consistent scene setup across all models

### 6.2 ✅ Diverse Probe Suite
- Tests multiple capability dimensions: spatial, linguistic, visual, attentional
- Good coverage of failure modes: perturbations, ablations, null cases
- Appropriate baseline comparisons

### 6.3 ✅ Proper Statistical Analysis
- DTW for trajectory comparison (handles temporal alignment)
- Correlation analysis (sensitivity vs displacement)
- Multi-seed variance measurement

## 7. Recommendations

### 7.1 Immediate Actions

1. **🚨 Debug OpenVLA determinism** 
   - Enable `do_sample=True` with temperature > 0
   - Verify action discretization/unnormalization pipeline
   - Check if model weights are frozen/cached

2. **⚠️ Validate π0 patches**
   - Test without compatibility shims on supported transformers version
   - Verify KV cache patch doesn't affect model outputs

3. **⚠️ Fix attention extraction**
   - Implement working attention hooks for π0/SmolVLA
   - Or exclude attention metrics for these models

### 7.2 Experimental Improvements

1. **Add sanity checks**:
   - Verify scene rendering matches expected setup
   - Check block positions after manipulations
   - Validate action bounds are reasonable

2. **Report domain gaps clearly**:
   - Flag π0/SmolVLA as cross-embodiment evaluation
   - Adjust interpretation/conclusions accordingly

3. **Expand OpenVLA testing**:
   - Test VLM querying capability (Probe 9)
   - Verify single-camera vs dual-camera impact

### 7.3 Long-term Methodology

1. **Standardize embodiments**: Use models' native training embodiments when possible
2. **Add ground-truth validation**: Compare against human demonstrations or real robot data  
3. **Expand metrics**: Add task-success metrics, safety measures, human preference alignment

## Conclusion

The VLA probing methodology is **fundamentally sound** with a robust evaluation framework and diverse probe suite. However, **critical implementation issues** undermine result validity:

- **OpenVLA exhibits completely deterministic behavior** (likely bug)
- **π0 requires fragile compatibility patches** (stability risk)
- **Cross-embodiment gaps** affect π0/SmolVLA interpretation
- **Attention extraction failed** for 2/4 models

**Overall Assessment**: 🟡 **CONDITIONALLY VALID**
- Results are meaningful for **X-VLA only**  
- **OpenVLA results should be discarded** until determinism is resolved
- **π0/SmolVLA results** need cross-embodiment caveats
- Framework is excellent once implementation issues are addressed

**Confidence in Results**: X-VLA (High), π0 (Medium), SmolVLA (Medium), OpenVLA (None)