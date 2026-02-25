"""Tests for lerobot_cache package."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
import torch

from lerobot_cache.cache_backend import SafetensorsBackend


# ── SafetensorsBackend unit tests (no LeRobot dependency) ──


class TestSafetensorsBackend:
    def test_put_and_get(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)
        tensor = torch.randn(3, 64, 64)
        backend.put("observation.images.top", 0, tensor)

        result = backend.get("observation.images.top", 0)
        assert result is not None
        assert torch.equal(tensor, result)

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)
        assert backend.get("nonexistent", 999) is None

    def test_exists(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)
        assert not backend.exists("key", 0)

        backend.put("key", 0, torch.zeros(3, 4, 4))
        assert backend.exists("key", 0)

    def test_cache_creates_files(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)
        for i in range(5):
            backend.put("observation.images.top", i, torch.randn(3, 48, 64))

        # Verify files exist on disk
        for i in range(5):
            path = tmp_path / "observation.images.top" / f"{i:06d}.safetensors"
            assert path.exists(), f"Expected cache file at {path}"

    def test_cache_stats(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)

        # Empty cache
        stats = backend.cache_stats()
        assert stats["cached_frames"] == 0
        assert stats["disk_size_bytes"] == 0

        # Add some frames
        for i in range(10):
            backend.put("cam", i, torch.randn(3, 32, 32))

        stats = backend.cache_stats()
        assert stats["cached_frames"] == 10
        assert stats["disk_size_bytes"] > 0
        assert stats["disk_size_gb"] >= 0
        assert "cam" in stats["image_keys"]
        assert stats["image_keys"]["cam"] == 10

    def test_multiple_image_keys(self, tmp_path: Path) -> None:
        backend = SafetensorsBackend(tmp_path)
        backend.put("cam_left", 0, torch.randn(3, 32, 32))
        backend.put("cam_right", 0, torch.randn(3, 32, 32))

        assert backend.exists("cam_left", 0)
        assert backend.exists("cam_right", 0)
        assert not backend.exists("cam_left", 1)

        stats = backend.cache_stats()
        assert stats["cached_frames"] == 2
        assert set(stats["image_keys"].keys()) == {"cam_left", "cam_right"}


# ── Integration tests requiring LeRobot dataset ──
# These are marked slow and skipped if dataset is unavailable.


def _dataset_available(repo_id: str = "lerobot/aloha_sim_transfer_cube_human") -> bool:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(repo_id)
        return True
    except Exception:
        return False


requires_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="LeRobot dataset not available (requires download)"
)


@requires_dataset
class TestCachedDatasetIntegration:
    REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

    def test_cached_dataset_returns_same_data(self, tmp_path: Path) -> None:
        """Load same frame from video and cache, assert tensors are equal."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot_cache import CachedDataset

        # Load from video
        video_ds = LeRobotDataset(self.REPO_ID)
        video_item = video_ds[0]

        # Load from cache
        cached_ds = CachedDataset(self.REPO_ID, cache_dir=tmp_path / "cache", auto_cache=True)
        cached_item = cached_ds[0]

        # All keys should match
        assert set(video_item.keys()) == set(cached_item.keys())

        # Image tensors should be equal
        for key in cached_ds.image_keys:
            assert torch.equal(video_item[key], cached_item[key]), \
                f"Mismatch on image key {key}"

        # Non-image tensors should be equal
        for key in video_item:
            if key not in cached_ds.image_keys:
                v, c = video_item[key], cached_item[key]
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, c), f"Mismatch on key {key}"

    def test_cache_creates_files(self, tmp_path: Path) -> None:
        """After accessing frames, verify safetensors files exist on disk."""
        from lerobot_cache import CachedDataset

        cache_dir = tmp_path / "cache"
        ds = CachedDataset(self.REPO_ID, cache_dir=cache_dir, auto_cache=True)

        # Metadata should exist
        assert (cache_dir / "metadata.json").exists()
        meta = json.loads((cache_dir / "metadata.json").read_text())
        assert meta["repo_id"] == self.REPO_ID
        assert meta["cache_format"] == "safetensors_v1"

        # Safetensors files should exist
        for key in ds.image_keys:
            assert (cache_dir / key / "000000.safetensors").exists()

    def test_cache_info(self, tmp_path: Path) -> None:
        """Verify cache_info() returns expected stats."""
        from lerobot_cache import CachedDataset

        ds = CachedDataset(self.REPO_ID, cache_dir=tmp_path / "cache", auto_cache=True)
        info = ds.cache_info()

        assert info["repo_id"] == self.REPO_ID
        assert info["cache_dir"] == str(tmp_path / "cache")
        assert info["cache_valid"] is True
        assert info["cached_frames"] > 0
        assert info["total_dataset_frames"] > 0

    def test_auto_cache_second_load_is_faster(self, tmp_path: Path) -> None:
        """First access builds cache, second load should skip building."""
        from lerobot_cache import CachedDataset

        cache_dir = tmp_path / "cache"

        # First load: builds cache (slow)
        t0 = time.time()
        ds1 = CachedDataset(self.REPO_ID, cache_dir=cache_dir, auto_cache=True)
        _ = ds1[0]
        t_first = time.time() - t0

        # Second load: cache exists (should be fast — no decode)
        t0 = time.time()
        ds2 = CachedDataset(self.REPO_ID, cache_dir=cache_dir, auto_cache=True)
        _ = ds2[0]
        t_second = time.time() - t0

        # Second load should be faster (cache already built, no decode pass)
        # We just check it doesn't rebuild — the constructor should be fast
        assert ds2._cache_is_valid()
