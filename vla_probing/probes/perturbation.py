"""Probe 8: Environment perturbation — move blocks, check trajectory adaptation.

Tests whether the model tracks object changes by moving blocks to
various positions and checking if predicted trajectories adapt accordingly.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import perturbation_sensitivity
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter
from vla_probing.scene import (
    RED_BLOCK_DEFAULT_POS,
    WidowXScene,
)
from vla_probing.tracking import ExperimentTracker

# Perturbation positions for the red block
PERTURBATION_POSITIONS = {
    "default": RED_BLOCK_DEFAULT_POS.copy(),
    "shifted_left": RED_BLOCK_DEFAULT_POS + np.array([0.0, 0.08, 0.0]),
    "shifted_right": RED_BLOCK_DEFAULT_POS + np.array([0.0, -0.08, 0.0]),
    "shifted_forward": RED_BLOCK_DEFAULT_POS + np.array([0.05, 0.0, 0.0]),
    "shifted_back": RED_BLOCK_DEFAULT_POS + np.array([-0.05, 0.0, 0.0]),
    "raised": RED_BLOCK_DEFAULT_POS + np.array([0.0, 0.0, 0.05]),
}


class PerturbationProbe(Probe):
    name = "perturbation"
    description = "Move blocks to test trajectory adaptation"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        prompt = "pick up the red block"

        # 1. Get baseline trajectory
        self.scene.reset()
        baseline_actions, baseline_views = self._predict(prompt)
        baseline_2d = np.atleast_2d(baseline_actions).reshape(
            -1, baseline_actions.shape[-1]
        )
        baseline_xyz = baseline_2d[:, :3]

        # 2. Run perturbations
        perturbation_results = {}
        for name, pos in PERTURBATION_POSITIONS.items():
            if name == "default":
                continue

            torch.manual_seed(seed)
            self.scene.reset()
            self.scene.set_red_block_pos(pos)
            actions, views = self._predict(prompt)
            actions_2d = np.atleast_2d(actions).reshape(-1, actions.shape[-1])
            traj_xyz = actions_2d[:, :3]

            # How much did the trajectory change?
            sensitivity = perturbation_sensitivity(baseline_2d, actions_2d)

            # Does the trajectory endpoint move toward the new block position?
            if len(traj_xyz) > 0:
                endpoint_to_block = float(np.linalg.norm(traj_xyz[-1] - pos))
            else:
                endpoint_to_block = float("inf")

            perturbation_results[name] = {
                "sensitivity": sensitivity,
                "endpoint_to_block": endpoint_to_block,
                "block_displacement": float(np.linalg.norm(pos - RED_BLOCK_DEFAULT_POS)),
                "traj_xyz": traj_xyz,
            }

        # Reset scene
        self.scene.reset()

        # Aggregate metrics
        sensitivities = [v["sensitivity"] for v in perturbation_results.values()]
        extra_metrics = {
            "mean_perturbation_sensitivity": float(np.mean(sensitivities)),
            "max_perturbation_sensitivity": float(np.max(sensitivities)),
        }

        for name, res in perturbation_results.items():
            extra_metrics[f"{name}_sensitivity"] = res["sensitivity"]
            extra_metrics[f"{name}_endpoint_dist"] = res["endpoint_to_block"]

        # Compute adaptation ratio: does trajectory shift proportionally to block shift?
        displacements = [v["block_displacement"] for v in perturbation_results.values()]
        if displacements:
            correlation = np.corrcoef(sensitivities, displacements)[0, 1]
            if not np.isnan(correlation):
                extra_metrics["sensitivity_displacement_correlation"] = float(
                    correlation
                )

        return self._make_result(
            variant="block_position_perturbations",
            actions=baseline_2d,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "baseline_view": baseline_views["image"],
                "baseline_xyz": baseline_xyz,
            },
        )


def main() -> None:
    parser = common_args("Probe 8: Environment perturbation")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"perturbation_{args.model}",
            config=vars(args),
            tags=[args.model, "perturbation"],
        )

    probe = PerturbationProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== Perturbation Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
