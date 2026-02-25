#!/usr/bin/env python
"""Pre-decode a LeRobot video dataset to cached safetensors files.

Decodes ALL frames from all image keys and saves each as a safetensors file.

Usage:
    python scripts/predecode_dataset.py lerobot/aloha_sim_transfer_cube_human --output outputs/cache/aloha
    python scripts/predecode_dataset.py lerobot/aloha_sim_transfer_cube_human  # uses default cache dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot_cache.cache_backend import SafetensorsBackend


def _default_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "lerobot-cache" / repo_id.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-decode LeRobot dataset to safetensors cache")
    parser.add_argument("repo_id", help="HuggingFace dataset repo ID")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output cache directory (default: ~/.cache/lerobot-cache/<dataset>)")
    args = parser.parse_args()

    repo_id = args.repo_id
    cache_dir = Path(args.output) if args.output else _default_cache_dir(repo_id)

    print(f"Pre-decoding dataset: {repo_id}")
    print(f"Output directory: {cache_dir}")
    print()

    # Load metadata
    meta = LeRobotDatasetMetadata(repo_id)
    image_keys = [
        key for key, feat in meta.features.items()
        if feat.get("dtype", "") in ("image", "video")
    ]

    if not image_keys:
        print("No image keys found in dataset. Nothing to decode.")
        return

    print(f"  Episodes:   {meta.total_episodes}")
    print(f"  Frames:     {meta.total_frames}")
    print(f"  FPS:        {meta.fps}")
    print(f"  Image keys: {image_keys}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id)
    n_frames = len(dataset)

    # Set up cache backend
    backend = SafetensorsBackend(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Decode all frames
    print(f"Decoding {n_frames} frames...")
    t_start = time.time()

    for idx in tqdm(range(n_frames), desc="Decoding"):
        item = dataset[idx]
        for key in image_keys:
            if key in item:
                backend.put(key, idx, item[key])

    elapsed = time.time() - t_start
    fps = n_frames / elapsed if elapsed > 0 else 0

    # Save metadata
    info = f"{meta.repo_id}:{meta.total_episodes}:{meta.total_frames}:{meta.fps}"
    metadata_hash = hashlib.sha256(info.encode()).hexdigest()[:16]

    metadata = {
        "repo_id": repo_id,
        "metadata_hash": metadata_hash,
        "image_keys": image_keys,
        "num_frames": n_frames,
        "cache_format": "safetensors_v1",
        "build_time_s": round(elapsed, 1),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Report
    stats = backend.cache_stats()
    print()
    print("=" * 50)
    print(f"Pre-decode complete!")
    print(f"  Frames:     {n_frames}")
    print(f"  Time:       {elapsed:.1f}s ({fps:.0f} frames/sec)")
    print(f"  Disk usage: {stats['disk_size_gb']:.2f} GB ({stats['disk_size_bytes'] / (1024**2):.0f} MB)")
    print(f"  Cache dir:  {cache_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
