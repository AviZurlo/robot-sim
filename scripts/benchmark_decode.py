#!/usr/bin/env python
"""Benchmark video decode vs cached frame loading for LeRobot datasets.

Compares sequential and random access patterns across:
  - Video decode (LeRobotDataset default)
  - Safetensors cache
  - NumPy memory-mapped files

Usage:
    python scripts/benchmark_decode.py
    python scripts/benchmark_decode.py --dataset lerobot/aloha_sim_transfer_cube_human --num-frames 500
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _get_image_keys(meta: LeRobotDatasetMetadata) -> list[str]:
    return [k for k, v in meta.features.items() if v.get("dtype", "") in ("image", "video")]


def benchmark_video_sequential(dataset: LeRobotDataset, image_key: str, indices: list[int]) -> float:
    """Time sequential frame access from video."""
    t0 = time.perf_counter()
    for idx in indices:
        _ = dataset[idx][image_key]
    return time.perf_counter() - t0


def benchmark_video_random(dataset: LeRobotDataset, image_key: str, indices: list[int]) -> float:
    """Time random-order frame access from video."""
    shuffled = list(indices)
    random.shuffle(shuffled)
    t0 = time.perf_counter()
    for idx in shuffled:
        _ = dataset[idx][image_key]
    return time.perf_counter() - t0


def benchmark_safetensors(cache_dir: Path, image_key: str, indices: list[int], random_order: bool) -> float:
    """Time frame loading from safetensors cache."""
    order = list(indices)
    if random_order:
        random.shuffle(order)
    t0 = time.perf_counter()
    for idx in order:
        path = cache_dir / image_key / f"{idx:06d}.safetensors"
        data = load_file(path)
        _ = data["frame"]
    return time.perf_counter() - t0


def benchmark_numpy_mmap(cache_dir: Path, image_key: str, indices: list[int], random_order: bool) -> float:
    """Time frame loading from numpy memory-mapped files."""
    order = list(indices)
    if random_order:
        random.shuffle(order)
    t0 = time.perf_counter()
    for idx in order:
        path = cache_dir / image_key / f"{idx:06d}.npy"
        arr = np.load(path, mmap_mode="r")
        _ = torch.from_numpy(np.array(arr))
    return time.perf_counter() - t0


def prepare_caches(
    dataset: LeRobotDataset, image_key: str, indices: list[int], tmp_dir: Path
) -> tuple[Path, Path]:
    """Decode frames and save as safetensors and numpy for benchmarking."""
    st_dir = tmp_dir / "safetensors"
    np_dir = tmp_dir / "numpy"
    (st_dir / image_key).mkdir(parents=True, exist_ok=True)
    (np_dir / image_key).mkdir(parents=True, exist_ok=True)

    print("  Preparing cache files for benchmark...")
    for idx in indices:
        tensor = dataset[idx][image_key]
        # Safetensors
        save_file({"frame": tensor}, st_dir / image_key / f"{idx:06d}.safetensors")
        # Numpy
        np.save(np_dir / image_key / f"{idx:06d}.npy", tensor.numpy())

    return st_dir, np_dir


def estimate_disk_cost(dataset: LeRobotDataset, image_key: str, sample_idx: int) -> dict:
    """Estimate full dataset cache disk cost from a single frame."""
    tensor = dataset[sample_idx][image_key]
    frame_bytes_st = len(torch.zeros(0).numpy().tobytes()) + tensor.nelement() * tensor.element_size() + 256  # safetensors overhead
    frame_bytes_np = tensor.nelement() * tensor.element_size() + 128  # npy overhead

    n_total = len(dataset)
    return {
        "frame_shape": list(tensor.shape),
        "frame_dtype": str(tensor.dtype),
        "safetensors_per_frame_kb": round(frame_bytes_st / 1024, 1),
        "numpy_per_frame_kb": round(frame_bytes_np / 1024, 1),
        "safetensors_total_gb": round(frame_bytes_st * n_total / (1024**3), 2),
        "numpy_total_gb": round(frame_bytes_np * n_total / (1024**3), 2),
        "total_frames": n_total,
    }


def run_benchmark(dataset_id: str, num_frames: int) -> None:
    """Run the full benchmark suite."""
    random.seed(42)

    print(f"Benchmark: {dataset_id}")
    print(f"  Frames to test: {num_frames}")
    print()

    # Load metadata
    meta = LeRobotDatasetMetadata(dataset_id)
    image_keys = _get_image_keys(meta)
    if not image_keys:
        print("ERROR: No image keys found in dataset. Nothing to benchmark.")
        return

    image_key = image_keys[0]
    print(f"  Using image key: {image_key}")
    print(f"  Dataset frames:  {meta.total_frames}")
    print(f"  Episodes:        {meta.total_episodes}")
    print(f"  FPS:             {meta.fps}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(dataset_id)
    n = min(num_frames, len(dataset))
    indices = list(range(n))

    # Estimate disk cost
    print("Estimating disk cost...")
    disk = estimate_disk_cost(dataset, image_key, 0)
    print(f"  Frame shape: {disk['frame_shape']} ({disk['frame_dtype']})")
    print(f"  Safetensors: ~{disk['safetensors_per_frame_kb']} KB/frame, ~{disk['safetensors_total_gb']} GB total")
    print(f"  NumPy:       ~{disk['numpy_per_frame_kb']} KB/frame, ~{disk['numpy_total_gb']} GB total")
    print()

    # Prepare cached versions
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        st_dir, np_dir = prepare_caches(dataset, image_key, indices, tmp_dir)

        # Warmup
        print("  Warmup (5 frames each)...")
        for idx in indices[:5]:
            _ = dataset[idx][image_key]
            load_file(st_dir / image_key / f"{idx:06d}.safetensors")
            np.load(np_dir / image_key / f"{idx:06d}.npy", mmap_mode="r")

        results = []

        # Video sequential
        print(f"  Benchmarking video decode (sequential, {n} frames)...")
        t = benchmark_video_sequential(dataset, image_key, indices)
        results.append(("Video decode", "sequential", n, t))

        # Video random
        print(f"  Benchmarking video decode (random, {n} frames)...")
        t = benchmark_video_random(dataset, image_key, indices)
        results.append(("Video decode", "random", n, t))

        # Safetensors sequential
        print(f"  Benchmarking safetensors (sequential, {n} frames)...")
        t = benchmark_safetensors(st_dir, image_key, indices, random_order=False)
        results.append(("Safetensors", "sequential", n, t))

        # Safetensors random
        print(f"  Benchmarking safetensors (random, {n} frames)...")
        t = benchmark_safetensors(st_dir, image_key, indices, random_order=True)
        results.append(("Safetensors", "random", n, t))

        # Numpy mmap sequential
        print(f"  Benchmarking numpy mmap (sequential, {n} frames)...")
        t = benchmark_numpy_mmap(np_dir, image_key, indices, random_order=False)
        results.append(("NumPy mmap", "sequential", n, t))

        # Numpy mmap random
        print(f"  Benchmarking numpy mmap (random, {n} frames)...")
        t = benchmark_numpy_mmap(np_dir, image_key, indices, random_order=True)
        results.append(("NumPy mmap", "random", n, t))

    # Print results table
    print()
    print("=" * 76)
    print(f"{'Backend':<18} {'Access':<12} {'Frames':>6} {'Time (s)':>9} {'ms/frame':>10} {'FPS':>10} {'Speedup':>8}")
    print("-" * 76)

    # Use video random as baseline for speedup
    video_random_time = None
    for name, access, frames, elapsed in results:
        if name == "Video decode" and access == "random":
            video_random_time = elapsed
            break

    for name, access, frames, elapsed in results:
        ms_per = (elapsed / frames) * 1000
        fps = frames / elapsed if elapsed > 0 else float("inf")
        if video_random_time and video_random_time > 0:
            speedup = video_random_time / elapsed
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "—"
        print(f"{name:<18} {access:<12} {frames:>6} {elapsed:>9.2f} {ms_per:>10.2f} {fps:>10.0f} {speedup_str:>8}")

    print("=" * 76)
    print()
    print(f"Baseline: video decode (random access)")
    print(f"Dataset: {dataset_id} ({meta.total_frames} frames, {meta.total_episodes} episodes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark video decode vs cached loading")
    parser.add_argument("--dataset", type=str, default="lerobot/aloha_sim_transfer_cube_human",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--num-frames", type=int, default=500, help="Number of frames to benchmark")
    args = parser.parse_args()

    run_benchmark(args.dataset, args.num_frames)


if __name__ == "__main__":
    main()
