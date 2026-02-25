"""CachedDataset — drop-in replacement for LeRobotDataset with frame caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot_cache.cache_backend import SafetensorsBackend


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "_")


def _metadata_hash(meta: LeRobotDatasetMetadata) -> str:
    """Hash dataset metadata for cache invalidation."""
    info = f"{meta.repo_id}:{meta.total_episodes}:{meta.total_frames}:{meta.fps}"
    return hashlib.sha256(info.encode()).hexdigest()[:16]


class CachedDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset, serving image frames from a safetensors cache.

    Non-image keys are passed through from the underlying dataset unchanged.
    """

    def __init__(
        self,
        repo_id: str,
        cache_dir: str | Path | None = None,
        auto_cache: bool = True,
        **kwargs,
    ) -> None:
        self.repo_id = repo_id

        # Load underlying dataset
        self.dataset = LeRobotDataset(repo_id, **kwargs)
        self.meta = self.dataset.meta

        # Resolve cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "lerobot-cache" / _sanitize_repo_id(repo_id)
        self.cache_dir = Path(cache_dir)

        # Identify image keys from dataset features
        self.image_keys = [
            key for key, feat in self.meta.features.items()
            if feat.get("dtype", "") == "image"
        ]

        self.backend = SafetensorsBackend(self.cache_dir)
        self._metadata_hash = _metadata_hash(self.meta)

        # Validate or build cache
        if self._cache_is_valid():
            pass  # Cache exists and matches
        elif auto_cache:
            self._build_cache()
        # else: no cache, will fall through to underlying dataset for images

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Replace image tensors with cached versions
        for key in self.image_keys:
            cached = self.backend.get(key, idx)
            if cached is not None:
                item[key] = cached

        return item

    def cache_info(self) -> dict:
        """Return cache status information."""
        stats = self.backend.cache_stats()
        stats["cache_dir"] = str(self.cache_dir)
        stats["repo_id"] = self.repo_id
        stats["image_keys"] = list(self.image_keys) if isinstance(stats.get("image_keys"), dict) else stats.get("image_keys", [])
        stats["total_dataset_frames"] = self.meta.total_frames
        stats["cache_valid"] = self._cache_is_valid()
        return stats

    def _cache_is_valid(self) -> bool:
        """Check if cache exists and matches dataset metadata."""
        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            return meta.get("metadata_hash") == self._metadata_hash
        except (json.JSONDecodeError, KeyError):
            return False

    def _build_cache(self) -> None:
        """Decode all frames and write to cache."""
        if not self.image_keys:
            return

        print(f"Building cache for {self.repo_id}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Image keys: {self.image_keys}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        n_frames = len(self.dataset)

        for idx in tqdm(range(n_frames), desc="Caching frames"):
            item = self.dataset[idx]
            for key in self.image_keys:
                if key in item:
                    self.backend.put(key, idx, item[key])

        elapsed = time.time() - t_start
        fps = n_frames / elapsed if elapsed > 0 else 0

        # Write metadata
        metadata = {
            "repo_id": self.repo_id,
            "metadata_hash": self._metadata_hash,
            "image_keys": self.image_keys,
            "num_frames": n_frames,
            "cache_format": "safetensors_v1",
            "build_time_s": round(elapsed, 1),
        }
        (self.cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        stats = self.backend.cache_stats()
        print(f"  Done: {n_frames} frames in {elapsed:.1f}s ({fps:.0f} fps)")
        print(f"  Disk usage: {stats['disk_size_gb']:.2f} GB")
