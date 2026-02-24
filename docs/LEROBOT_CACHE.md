# lerobot-cache — Training Data Optimization for LeRobot

**Status:** Planned (sub-project of robot-sim)
**Created:** 2026-02-24
**Goal:** Open-source tool to eliminate video decoding bottleneck in LeRobot training pipelines

## The Problem

LeRobot stores robot demonstration datasets as compressed MP4 videos. During training, every batch requires:

1. Opening the MP4 file
2. Seeking to the correct timestamp
3. CPU-decoding compressed frames to raw pixels
4. Converting to PyTorch tensors
5. Feeding through the vision encoder

Steps 1-3 happen **every batch, every epoch**, and are entirely CPU-bound. On hardware without NVIDIA's nvdec (Mac, AMD, CPU-only servers), video decoding becomes the dominant training bottleneck — not the actual model forward/backward pass.

### Evidence

- **Our experience:** 51M param ACT model on Mac mini couldn't complete a single training step in 20+ minutes. The model itself is fine — PushT (state-only, no video) trains at 4.2 steps/sec on the same hardware.
- **GitHub Issue #436:** Community member asked about image storage formats for large-scale training. Marked "not planned" by LeRobot team.
- **GitHub PR #1671:** GPU-accelerated video encoding (recording side) — confirms the team knows video I/O is a bottleneck, but only addresses the recording direction.
- **`any4lerobot` (865★):** Community utility collection with data conversion scripts, but no transparent caching layer.

### Who is affected

- Anyone training vision-based policies on non-NVIDIA hardware (Mac, AMD, cloud CPU instances)
- Anyone training on large datasets where decode time exceeds compute time
- Hobbyists and researchers with consumer hardware (the growing LeRobot community)

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

| Solution | What it does | Gap |
|----------|-------------|-----|
| LeRobot built-in `.pt` storage | Stores tensors instead of video | Not well-documented, no conversion CLI, no caching |
| `any4lerobot` (865★) | Data conversion utilities | Grab bag, not a focused caching layer |
| PR #1671 (GPU async encoding) | Faster video recording | Recording side only, doesn't help training |
| NVIDIA nvdec | Hardware video decoding | NVIDIA-only, not available on Mac/AMD/CPU |
| Manual pre-decoding | Convert videos to images yourself | Everyone reinvents this, no standard tool |
| **lerobot-cache (ours)** | **Transparent caching + CLI + benchmarks** | **This is what we'd build** |

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

- Should this be a standalone package (`lerobot-cache`) or a PR to LeRobot itself?
  - Leaning standalone first (faster iteration, no review bottleneck), then upstream if adopted
- What cache format? Safetensors (HF ecosystem), numpy mmap (fastest), or both?
- Should we support partial caching (only cache frequently-accessed frames)?
- Is VideoToolbox (Phase 4) worth the complexity, or is disk caching sufficient?

## Resources

- LeRobot dataset internals: `lerobot/common/datasets/lerobot_dataset.py`
- LeRobot video utils: `lerobot/common/datasets/video_utils.py`
- Issue #436: Image storage format discussion
- `any4lerobot`: https://github.com/Tavish9/any4lerobot
- Diffusion Policy paper (PushT benchmark): https://arxiv.org/abs/2303.04137
- VideoToolbox docs: https://developer.apple.com/documentation/videotoolbox
