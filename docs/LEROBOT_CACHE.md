# lerobot-cache — Training Data Optimization for LeRobot

**Status:** Research complete, ready for Phase 1 (benchmarking)
**Created:** 2026-02-24
**Last Updated:** 2026-02-24
**Goal:** Open-source tool to eliminate video decoding bottleneck in LeRobot training pipelines

---

## The Problem (Plain English)

Imagine you have a textbook, but it's written in a secret code. Every time you want to study a page, you have to decode it first — letter by letter. And the next time you study, you decode it all over again. That's how LeRobot trains robots right now.

**LeRobot stores training data as compressed MP4 videos.** These are great for saving disk space and sharing datasets online. But during training, your computer has to "unzip" every video frame on the fly — thousands of times per training run. This video decoding is slow, CPU-intensive work, and it happens over and over again even though the frames never change.

**The fix is simple in concept:** decode the videos once, save the decoded frames, and load those instead. Like translating the textbook into plain English once and studying from the translation forever after. That's what lerobot-cache does.

### Why This Matters (Numbers)

On our Mac mini (Apple Silicon, 32GB RAM):
- **PushT (no video):** 4.2 training steps/second ✅
- **ALOHA ACT (with video):** Couldn't finish a single step in 20+ minutes ❌
- **Same model complexity, same hardware.** The ONLY difference is video decoding.

This isn't a Mac-specific problem — it affects anyone without a dedicated hardware video decoder chip (NVIDIA's nvdec), including AMD GPUs, cloud CPU instances, and the entire hobbyist community.

### How Training Works Today (The Slow Way)

Every training batch requires:
1. Opening the MP4 file
2. Seeking to the correct timestamp
3. CPU-decoding compressed frames to raw pixels
4. Converting to PyTorch tensors
5. Feeding through the vision encoder

Steps 1-3 happen **every batch, every epoch**, and are entirely CPU-bound. The GPU sits idle waiting for data. It's like having a Ferrari stuck behind a horse-drawn carriage.

### How lerobot-cache Would Work (The Fast Way)

1. **First time:** Decode all video frames once → save as ready-to-use image tensors on disk
2. **Every subsequent training run:** Load pre-decoded tensors directly → skip video decoding entirely
3. **Result:** Training speed is limited by your GPU/model, not your video decoder

---

## Upstream Research Findings

We dug deep into the LeRobot community to understand what others have tried and what gaps exist.

### Issue #436 — "Image Storage Format" (Closed, Oct 2025)

**What it was:** nikonikolov asked the LeRobot team about image storage options — comparing raw images vs compressed video for training data.

**Key findings from discussion:**
- **Cadene (LeRobot maintainer)** said PNG vs MP4 showed no quality difference for their benchmarks, and preferred video for smaller dataset sizes on Hugging Face Hub
- **richardrl reported a ~20x speedup** using FFCV (Fast Forward Computer Vision) — a sharded binary format that stores pre-decoded images. This was on ~20TB of real-world robot data over NFS storage
- Issue was closed as "not planned" — the team is aware of the tradeoff but chose disk-size efficiency over training speed

**What this means for us:** The ~20x speedup from richardrl is real-world validation that pre-decoding works at scale. FFCV is the closest existing approach to what we're building, but it's a general computer vision tool — not tailored to LeRobot's specific dataset format or workflow.

### Issue #1281 — "Large Scale Training" (Open, No Response)

**What it is:** richardrl (same person who reported the FFCV speedup) followed up asking the LeRobot team directly about throughput benchmarks for large-scale training. Specifically asked about:
- Storage requirements for Parquet (LeRobot's newer format) vs video
- Throughput numbers for the largest datasets trained with LeRobot
- How Parquet compares to webdataset and torchcodec (on-the-fly MP4 decoding)

**Status:** No response from the LeRobot team as of Feb 2026. This tells us:
1. The team hasn't published benchmarks for large-scale training throughput
2. There's clear community demand for this data
3. If we build lerobot-cache with solid benchmarks, we'd be filling an obvious gap

### PR #1671 — GPU-Accelerated Async Video Encoding (Open)

**What it does:** Adds NVIDIA NVENC support for faster video **recording** (i.e., when you're collecting new robot demonstration data). Claims 3-4x speedup for encoding.

**Why it doesn't solve our problem:** This speeds up *saving* data, not *loading* it for training. It's like having a faster printer — helpful, but doesn't make reading faster. Also requires NVIDIA GPU, so no help on Mac/AMD.

**What it tells us:** The LeRobot team and community recognize video I/O as a bottleneck. But effort has focused on the recording side, leaving the training side unaddressed.

### `any4lerobot` (865★ Community Tool)

**What it is:** A grab-bag of community utilities for LeRobot — data conversion scripts, format helpers, various quality-of-life tools.

**What it lacks:** No transparent caching layer. No training-side optimization. Not a competitor to lerobot-cache — more like a toolbox that our tool could complement.

### LeRobot's Own Format Evolution

LeRobot has been evolving its data format:
- **v1.x:** Pure MP4 video + JSON metadata
- **v2.x:** Parquet files for tabular data (actions, states) + MP4 for images
- The move to Parquet shows the team cares about data loading speed for non-image data, but images are still video-encoded

---

## How We Compare to Existing Solutions

| Approach | What It Does | Training Speedup | LeRobot-Specific? | Drop-in? | Status |
|----------|-------------|------------------|--------------------|----------|--------|
| **LeRobot default** | Decode MP4 on every batch | Baseline (1x) | ✅ | ✅ | Current |
| **FFCV** | Pre-decoded sharded binary format | ~20x (reported) | ❌ General CV tool | ❌ Requires rewrite | Active project |
| **webdataset** | Sharded tar archives for distributed training | 2-5x (estimated) | ❌ General ML tool | ❌ Different format | Active project |
| **torchcodec** | Optimized video decoding (still on-the-fly) | 1.5-2x (estimated) | ❌ General tool | ⚠️ Partial | PyTorch project |
| **PR #1671** | GPU-accelerated recording | 0x (recording only) | ✅ | ✅ | Open PR |
| **any4lerobot** | Data conversion utilities | 0x (no caching) | ✅ | ❌ | Community tool |
| **Manual pre-decoding** | DIY: convert videos to images | 5-20x (depends) | ❌ Everyone reinvents it | ❌ | Common hack |
| **lerobot-cache (ours)** | **Transparent cache + CLI + benchmarks** | **5-20x (target)** | **✅ Built for LeRobot** | **✅ Drop-in** | **Planned** |

### Our Unique Value

1. **LeRobot-native:** Works with LeRobot's dataset format, APIs, and training scripts — not a generic tool that requires adaptation
2. **Drop-in replacement:** One line of code to switch from slow to fast — `CachedDataset(...)` instead of `LeRobotDataset(...)`
3. **Benchmarked:** We'll ship with real numbers on real hardware (Mac, NVIDIA, CPU-only) — something the community is explicitly asking for (issue #1281)
4. **CLI for non-coders:** `lerobot-cache prepare <dataset>` — hobbyists shouldn't need to write Python to make training faster
5. **Community-facing:** Standalone PyPI package first for fast iteration, then propose upstream integration if adopted

## Proposed Solution

A transparent caching layer that sits between LeRobot's dataset and the training loop. Decode once, train fast forever.

### Core Design

```python
# Before (slow — decodes video every batch)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

# After (fast — decodes once, caches to disk)
from lerobot_cache import CachedDataset
dataset = CachedDataset("lerobot/aloha_sim_transfer_cube_human")
# First epoch: decode + cache. All subsequent: load cached tensors.
```

### Key Properties

- **Drop-in replacement:** Works with any LeRobot dataset, any policy, any training script
- **Lazy caching:** Frames decoded and cached on first access — no upfront preprocessing required
- **Configurable backends:** Disk (safetensors), memory-mapped (numpy mmap), RAM (for small datasets)
- **Cache management:** LRU eviction with configurable max size, cache info/stats
- **Data integrity:** Cached data must produce identical training results to video decoding (bit-for-bit tensor equality)

## Build Plan

### Phase 1: Prove the Speedup (1-2 days)

**Goal:** Hard benchmark numbers before writing any fancy code.

**Tasks:**
1. Write `scripts/benchmark_decode.py` — measures wall-clock time per batch for video vs pre-decoded data
2. Build a simple pre-decode script that converts ALOHA video dataset to image tensors on disk
3. Modify `train.py` to optionally load from pre-decoded data
4. Run identical training (same seed, same hyperparams) with both paths
5. Compare: steps/sec, wall-clock time, GPU/CPU utilization, and verify identical loss curves

**Target metrics:**
- 5-10x speedup on Mac mini (Apple Silicon, no hardware video decoder for ML)
- 2-3x speedup on NVIDIA (where nvdec helps but decode is still not free)
- Identical training loss curves (proves no data quality degradation)

**Hardware for benchmarks:**
- Mac mini M-series 32GB (our machine)
- NVIDIA A100 (one-off cloud GPU via Modal/Lambda, ~$2)
- CPU-only Linux (GitHub Actions or similar)

### Phase 2: Drop-in Caching Wrapper (3-5 days)

**Goal:** `CachedLeRobotDataset` class that wraps any `LeRobotDataset`.

**Implementation:**
```python
class CachedLeRobotDataset:
    """Transparent caching wrapper for LeRobot datasets.
    
    First access decodes video frames and saves to cache directory.
    Subsequent accesses load directly from cache (no video decoding).
    
    Cache backends:
    - 'disk': Safetensors files (default, good balance of speed and space)
    - 'mmap': Memory-mapped numpy arrays (fastest random access)
    - 'ram': In-memory dict (fastest, but limited by available RAM)
    """
    
    def __init__(self, repo_id, cache_dir=None, backend='disk', 
                 max_cache_gb=50, **kwargs):
        self.dataset = LeRobotDataset(repo_id, **kwargs)
        self.cache = CacheBackend(backend, cache_dir, max_cache_gb)
    
    def __getitem__(self, idx):
        cached = self.cache.get(idx)
        if cached is not None:
            return cached
        item = self.dataset[idx]
        self.cache.put(idx, item)
        return item
```

**Key design decisions:**
- Cache key = dataset repo_id + index (handles dataset versioning)
- Cache invalidation: hash of dataset metadata (if dataset updates, cache auto-invalidates)
- Thread-safe for multi-worker DataLoader
- Graceful fallback: if cache fails, silently falls back to video decoding

### Phase 3: CLI + Benchmarking (2-3 days)

**Goal:** User-facing CLI for dataset optimization and performance measurement.

**Commands:**
```bash
# Pre-decode entire dataset (upfront, one-time)
lerobot-cache prepare lerobot/aloha_sim_transfer_cube_human

# Benchmark decode vs cached performance
lerobot-cache benchmark lerobot/aloha_sim_transfer_cube_human
# Output: "Video decode: 0.8 steps/s | Cached: 6.2 steps/s | Speedup: 7.8x"

# Show cache status
lerobot-cache info
# Output: "Cached datasets: 2 | Total size: 12.3 GB | Estimated speedup: 5-8x"

# Clear cache for a dataset
lerobot-cache clear lerobot/aloha_sim_transfer_cube_human
```

**README requirements:**
- Benchmark table with before/after numbers on 3 hardware configs
- One-line install: `pip install lerobot-cache`
- Integration examples for common training scripts
- FAQ: cache size estimates per dataset, data quality guarantees

### Phase 4: Apple Silicon Optimization (Stretch Goal)

**Goal:** Use Apple's VideoToolbox hardware decoder instead of CPU ffmpeg.

Every Mac has a dedicated video decode chip that's currently unused during training. VideoToolbox can decode H.264/H.265 at ~4K60fps with near-zero CPU usage. If we can pipe decoded frames directly into PyTorch tensors, this could make video-based training competitive on Apple Silicon without any caching.

**This would be unique** — no existing PyTorch dataloader uses VideoToolbox.

**Implementation sketch:**
- Python bindings to VideoToolbox via `ctypes` or `pyobjc`
- Custom `torchcodec` backend that uses VT instead of ffmpeg
- Frames decoded on Apple Media Engine → copied to shared memory → PyTorch tensor

**Risk:** VideoToolbox API is Objective-C, bridging to Python adds complexity. May not be worth it if disk caching is already fast enough.

## Competitive Landscape

*(Detailed comparison table is in the "How We Compare" section above)*

## Testing Plan (Before Shipping)

### Correctness Tests
1. **Bit-for-bit equivalence:** Decode frame N from video and from cache → assert tensor equality
2. **Training reproducibility:** Same seed + same data → identical loss at every step, with and without caching
3. **Cache invalidation:** Update dataset on HF Hub → verify cache auto-invalidates and re-decodes

### Performance Tests
1. **Benchmark matrix:** 3 hardware configs × 3 datasets × video vs cached
2. **Memory profiling:** Caching shouldn't OOM on 8GB machines
3. **Multi-worker safety:** 4 DataLoader workers accessing cache simultaneously
4. **Cache cold start:** First epoch (decode + cache) should be ≤1.5x slower than video-only (the caching overhead shouldn't be worse than decoding)

### Datasets to Test
1. `lerobot/aloha_sim_transfer_cube_human` — 50 episodes, 2 cameras, standard benchmark
2. `lerobot/pusht` — state + video versions (control comparison)
3. A real-world dataset with many episodes (stress test cache management)

## Success Criteria

1. **Measurable speedup:** ≥5x on Mac, ≥2x on NVIDIA, with published benchmarks
2. **Zero code changes:** Works as a drop-in replacement for LeRobotDataset
3. **Community adoption:** Published on PyPI, linked from LeRobot discussions
4. **Potential upstream PR:** If the approach proves out, propose integration into LeRobot core

## Open Questions

- ~~Should this be a standalone package or a PR to LeRobot?~~ **Decided: Standalone first.** Faster iteration, and we can propose upstream after proving value.
- What cache format? Safetensors (HF ecosystem), numpy mmap (fastest), or both? → **Test both in Phase 1 benchmarks**
- Should we support partial caching (only cache frequently-accessed frames)? → Probably not for v1; full decode is simpler and disk is cheap
- Is VideoToolbox (Phase 4) worth the complexity, or is disk caching sufficient? → **Benchmark first, decide after**
- Should we respond to issue #1281 with our benchmark results once we have them? → Yes — this would be the perfect introduction to the community

## Resources

- LeRobot dataset internals: `lerobot/common/datasets/lerobot_dataset.py`
- LeRobot video utils: `lerobot/common/datasets/video_utils.py`
- **Issue #436:** Image storage format discussion — https://github.com/huggingface/lerobot/issues/436 (closed, "not planned")
- **Issue #1281:** Large scale training throughput — https://github.com/huggingface/lerobot/issues/1281 (open, unanswered)
- **PR #1671:** GPU async video encoding — https://github.com/huggingface/lerobot/pull/1671 (open, recording-side only)
- **FFCV:** https://github.com/libffcv/ffcv — general CV data loading library (richardrl reported ~20x speedup)
- `any4lerobot`: https://github.com/Tavish9/any4lerobot (865★ community tools)
- Diffusion Policy paper (PushT benchmark): https://arxiv.org/abs/2303.04137
- VideoToolbox docs: https://developer.apple.com/documentation/videotoolbox
