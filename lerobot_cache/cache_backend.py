"""Safetensors-based cache backend for decoded video frames."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class SafetensorsBackend:
    """Read/write individual frames as safetensors files.

    Layout: {cache_dir}/{image_key}/{frame_index:06d}.safetensors
    Each file contains a single tensor under the key "frame".
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def get(self, image_key: str, index: int) -> torch.Tensor | None:
        """Load a cached frame tensor, or None if not cached."""
        path = self._path(image_key, index)
        if not path.exists():
            return None
        data = load_file(path)
        return data["frame"]

    def put(self, image_key: str, index: int, tensor: torch.Tensor) -> None:
        """Save a frame tensor to cache."""
        path = self._path(image_key, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"frame": tensor}, path)

    def exists(self, image_key: str, index: int) -> bool:
        """Check if a cached frame exists."""
        return self._path(image_key, index).exists()

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        total_files = 0
        total_bytes = 0
        keys: dict[str, int] = {}

        if not self.cache_dir.exists():
            return {"cached_frames": 0, "disk_size_bytes": 0, "disk_size_gb": 0.0, "image_keys": {}}

        for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                if f.endswith(".safetensors"):
                    total_files += 1
                    total_bytes += (Path(dirpath) / f).stat().st_size
                    # Parent dir name is the image key
                    key = Path(dirpath).name
                    keys[key] = keys.get(key, 0) + 1

        return {
            "cached_frames": total_files,
            "disk_size_bytes": total_bytes,
            "disk_size_gb": round(total_bytes / (1024**3), 3),
            "image_keys": keys,
        }

    def _path(self, image_key: str, index: int) -> Path:
        # Sanitize image key (e.g. "observation.images.top" -> "observation.images.top")
        return self.cache_dir / image_key / f"{index:06d}.safetensors"
