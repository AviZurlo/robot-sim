"""CLI entry point for lerobot-cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot_cache.cache_backend import SafetensorsBackend


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "_")


def _default_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "lerobot-cache" / _sanitize_repo_id(repo_id)


def cmd_prepare(args: argparse.Namespace) -> None:
    """Pre-decode a dataset to safetensors cache."""
    repo_id = args.repo_id
    cache_dir = Path(args.output) if args.output else _default_cache_dir(repo_id)

    print(f"Preparing cache for {repo_id}")
    print(f"  Output: {cache_dir}")

    meta = LeRobotDatasetMetadata(repo_id)
    image_keys = [
        key for key, feat in meta.features.items()
        if feat.get("dtype", "") in ("image", "video")
    ]

    if not image_keys:
        print("  No image keys found — nothing to cache.")
        return

    print(f"  Image keys: {image_keys}")
    print(f"  Total frames: {meta.total_frames}")
    print(f"  Episodes: {meta.total_episodes}")
    print()

    dataset = LeRobotDataset(repo_id)
    backend = SafetensorsBackend(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    n_frames = len(dataset)

    for idx in tqdm(range(n_frames), desc="Decoding frames"):
        item = dataset[idx]
        for key in image_keys:
            if key in item:
                backend.put(key, idx, item[key])

    elapsed = time.time() - t_start
    fps = n_frames / elapsed if elapsed > 0 else 0
    stats = backend.cache_stats()

    # Save metadata
    import hashlib
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

    print()
    print(f"Done: {n_frames} frames in {elapsed:.1f}s ({fps:.0f} fps)")
    print(f"Disk usage: {stats['disk_size_gb']:.2f} GB")
    print(f"Cache dir: {cache_dir}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run decode benchmark (delegates to benchmark script)."""
    from scripts.benchmark_decode import run_benchmark
    run_benchmark(args.repo_id, args.num_frames)


def cmd_info(args: argparse.Namespace) -> None:
    """Show cache info."""
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path.home() / ".cache" / "lerobot-cache"

    if not cache_dir.exists():
        print(f"No cache found at {cache_dir}")
        return

    # If pointing at a specific dataset cache
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        backend = SafetensorsBackend(cache_dir)
        stats = backend.cache_stats()
        print(f"Cache: {cache_dir}")
        print(f"  Dataset:  {meta.get('repo_id', 'unknown')}")
        print(f"  Frames:   {stats['cached_frames']}")
        print(f"  Disk:     {stats['disk_size_gb']:.2f} GB")
        print(f"  Keys:     {meta.get('image_keys', [])}")
        print(f"  Format:   {meta.get('cache_format', 'unknown')}")
        return

    # List all cached datasets
    found = False
    for subdir in sorted(cache_dir.iterdir()):
        sub_meta = subdir / "metadata.json"
        if sub_meta.exists():
            found = True
            meta = json.loads(sub_meta.read_text())
            backend = SafetensorsBackend(subdir)
            stats = backend.cache_stats()
            print(f"{meta.get('repo_id', subdir.name)}")
            print(f"  Frames: {stats['cached_frames']}  Disk: {stats['disk_size_gb']:.2f} GB")

    if not found:
        print(f"No cached datasets found in {cache_dir}")


def cmd_clear(args: argparse.Namespace) -> None:
    """Delete cache for a dataset."""
    repo_id = args.repo_id
    cache_dir = _default_cache_dir(repo_id)

    if not cache_dir.exists():
        print(f"No cache found for {repo_id} at {cache_dir}")
        return

    backend = SafetensorsBackend(cache_dir)
    stats = backend.cache_stats()
    print(f"Deleting cache for {repo_id}")
    print(f"  {stats['cached_frames']} frames, {stats['disk_size_gb']:.2f} GB")

    shutil.rmtree(cache_dir)
    print("  Deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lerobot-cache", description="Cache layer for LeRobot datasets")
    sub = parser.add_subparsers(dest="command")

    # prepare
    p_prepare = sub.add_parser("prepare", help="Pre-decode dataset to cache")
    p_prepare.add_argument("repo_id", help="HuggingFace dataset repo ID")
    p_prepare.add_argument("--output", "-o", help="Output cache directory")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run decode benchmark")
    p_bench.add_argument("repo_id", help="HuggingFace dataset repo ID")
    p_bench.add_argument("--num-frames", "-n", type=int, default=500, help="Frames to benchmark")

    # info
    p_info = sub.add_parser("info", help="Show cache status")
    p_info.add_argument("--cache-dir", help="Cache directory to inspect")

    # clear
    p_clear = sub.add_parser("clear", help="Delete cache for dataset")
    p_clear.add_argument("repo_id", help="HuggingFace dataset repo ID")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "prepare": cmd_prepare,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
        "clear": cmd_clear,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
