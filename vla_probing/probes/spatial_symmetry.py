"""Probe 2: Spatial symmetry — swap block positions, compare trajectories.

Tests whether the model understands absolute vs relative positions
by swapping the red and blue blocks and comparing the predicted trajectories.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import perturbation_sensitivity, trajectory_dtw
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker


class SpatialSymmetryProbe(Probe):
    name = "spatial_symmetry"
    description = "Swap block positions to test spatial understanding"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        prompt = "pick up the red block"

        # 1. Baseline prediction (default positions)
        self.scene.reset()
        baseline_actions, baseline_views = self._predict(prompt)
        baseline_2d = np.atleast_2d(baseline_actions).reshape(
            -1, baseline_actions.shape[-1]
        )
        baseline_xyz = baseline_2d[:, :3]

        # 2. Swapped prediction
        self.scene.reset()
        self.scene.swap_block_positions()
        swapped_actions, swapped_views = self._predict(prompt)
        swapped_2d = np.atleast_2d(swapped_actions).reshape(
            -1, swapped_actions.shape[-1]
        )
        swapped_xyz = swapped_2d[:, :3]

        # Reset scene back to default
        self.scene.reset()

        # Metrics
        extra_metrics = {
            "perturbation_sensitivity": perturbation_sensitivity(
                baseline_2d, swapped_2d
            ),
        }

        # DTW between baseline and swapped trajectories
        if len(baseline_xyz) >= 2 and len(swapped_xyz) >= 2:
            extra_metrics["dtw_baseline_vs_swapped"] = trajectory_dtw(
                baseline_xyz, swapped_xyz
            )

        # Check if swapped trajectory correctly redirects to new red block position
        red_pos_swapped = self.scene.get_block_pos("red")  # after reset, back to original
        # The swapped red block was at blue's original position
        from vla_probing.scene import BLUE_BLOCK_DEFAULT_POS

        if len(swapped_xyz) > 0:
            extra_metrics["swapped_distance_to_original_blue_pos"] = float(
                np.linalg.norm(swapped_xyz[-1] - BLUE_BLOCK_DEFAULT_POS)
            )

        # Log both trajectories
        self.tracker.log_trajectory_plot(baseline_xyz, "spatial_baseline")
        self.tracker.log_trajectory_plot(swapped_xyz, "spatial_swapped")

        return self._make_result(
            variant="swap_red_blue",
            actions=swapped_2d,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "baseline_view": baseline_views["image"],
                "swapped_view": swapped_views["image"],
                "baseline_xyz": baseline_xyz,
                "swapped_xyz": swapped_xyz,
            },
        )


def main() -> None:
    parser = common_args("Probe 2: Spatial symmetry")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"spatial_symmetry_{args.model}",
            config=vars(args),
            tags=[args.model, "spatial_symmetry"],
        )

    probe = SpatialSymmetryProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== Spatial Symmetry Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
