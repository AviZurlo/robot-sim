"""Base probe infrastructure shared by all probe implementations."""

import argparse
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from vla_probing.adapter import VLAAdapter, VLAInput, XVLAAdapter
from vla_probing.adapters.openvla import OpenVLAAdapter
from vla_probing.adapters.pi0 import Pi0Adapter
from vla_probing.adapters.smolvla import SmolVLAAdapter
from vla_probing.metrics import compute_all_metrics
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker, ProbeResult


class Probe(ABC):
    """Base class for VLA diagnostic probes."""

    name: str  # e.g. "baseline", "spatial_symmetry"
    description: str

    def __init__(
        self,
        adapter: VLAAdapter,
        scene: WidowXScene,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.adapter = adapter
        self.scene = scene
        self.tracker = tracker or ExperimentTracker(enabled=False)

    @abstractmethod
    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        """Execute the probe and return results."""

    def _predict(
        self,
        prompt: str,
        scene: WidowXScene | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run VLA prediction on the current scene.

        Returns (actions, views) tuple.
        """
        s = scene or self.scene
        views = s.render_all_views()
        ee_state = s.get_ee_state()

        inp = VLAInput(
            images=[views["image"], views["image2"]],
            prompt=prompt,
            proprio=ee_state,
        )

        self.adapter.reset()
        output = self.adapter.predict_action(inp)
        return output.actions, views

    def _predict_multi_seed(
        self,
        prompt: str,
        n_seeds: int = 10,
    ) -> list[np.ndarray]:
        """Run VLA prediction with multiple seeds for stochasticity analysis."""
        import torch

        views = self.scene.render_all_views()
        ee_state = self.scene.get_ee_state()
        inp = VLAInput(
            images=[views["image"], views["image2"]],
            prompt=prompt,
            proprio=ee_state,
        )

        results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            self.adapter.reset()
            output = self.adapter.predict_action(inp)
            results.append(output.actions)
        return results

    def _make_result(
        self,
        variant: str,
        actions: np.ndarray,
        seed: int = 0,
        extra_metrics: dict[str, float] | None = None,
        artifacts: dict | None = None,
    ) -> ProbeResult:
        """Build a standardized ProbeResult."""
        metrics = compute_all_metrics(actions)
        if extra_metrics:
            metrics.update(extra_metrics)

        result = ProbeResult(
            model=self.adapter.model_name,
            embodiment="widowx",
            probe=self.name,
            probe_variant=variant,
            seed=seed,
            metrics=metrics,
            artifacts=artifacts or {},
        )

        self.tracker.log_probe_result(result)
        return result


def make_adapter(model: str = "xvla", device: str = "mps") -> VLAAdapter:
    """Factory to create a VLA adapter by model name."""
    adapters = {
        "xvla": XVLAAdapter,
        "pi0": Pi0Adapter,
        "smolvla": SmolVLAAdapter,
        "openvla": OpenVLAAdapter,
    }
    if model not in adapters:
        raise ValueError(f"Unknown model: {model}. Available: {list(adapters.keys())}")
    adapter = adapters[model]()
    adapter.load_model(device=device)
    return adapter


def common_args(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common probe arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model", default="xvla", choices=["xvla", "pi0", "smolvla", "openvla"], help="VLA model to probe"
    )
    parser.add_argument(
        "--device", default="mps", choices=["mps", "cpu", "cuda"], help="Device"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable W&B logging"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10, help="Number of seeds for stochasticity"
    )
    return parser
