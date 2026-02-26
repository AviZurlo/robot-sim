"""Probe 1: Baseline trajectory — 'pick up the red block' action visualization.

Tests whether the model reaches for the correct object and produces
reasonable action trajectories from the default scene configuration.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import trajectory_spread
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker


class BaselineProbe(Probe):
    name = "baseline"
    description = "Baseline trajectory for 'pick up the red block'"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        prompt = "pick up the red block"
        n_seeds = kwargs.get("n_seeds", 10)

        # Single prediction for primary result
        import torch

        torch.manual_seed(seed)
        actions, views = self._predict(prompt)

        # Extract XYZ trajectory
        actions_2d = np.atleast_2d(actions)
        if actions_2d.ndim == 3:
            # (batch, chunk, action_dim) -> flatten
            actions_2d = actions_2d.reshape(-1, actions_2d.shape[-1])
        traj_xyz = actions_2d[:, :3]

        # Check if trajectory points toward red block
        red_pos = self.scene.get_block_pos("red")
        extra_metrics = {}
        if red_pos is not None and len(traj_xyz) > 0:
            # Direction alignment: how much does the trajectory move toward the target?
            start_to_target = red_pos - traj_xyz[0]
            start_to_end = traj_xyz[-1] - traj_xyz[0]
            dist = np.linalg.norm(start_to_target)
            if dist > 1e-6 and np.linalg.norm(start_to_end) > 1e-6:
                alignment = np.dot(
                    start_to_target / np.linalg.norm(start_to_target),
                    start_to_end / np.linalg.norm(start_to_end),
                )
                extra_metrics["direction_alignment"] = float(alignment)
            extra_metrics["distance_to_target"] = float(
                np.linalg.norm(traj_xyz[-1] - red_pos)
            )

        # Multi-seed stochasticity analysis
        all_trajs = self._predict_multi_seed(prompt, n_seeds=n_seeds)
        all_xyz = [np.atleast_2d(t).reshape(-1, t.shape[-1])[:, :3] for t in all_trajs]
        extra_metrics["trajectory_spread"] = trajectory_spread(all_xyz)

        # Log trajectory plot
        self.tracker.log_trajectory_plot(traj_xyz, name="baseline_trajectory")

        return self._make_result(
            variant="pick_up_red_block",
            actions=actions_2d,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "primary_view": views["image"],
                "side_view": views["image2"],
                "trajectory_xyz": traj_xyz,
            },
        )


def main() -> None:
    parser = common_args("Probe 1: Baseline trajectory")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"baseline_{args.model}",
            config=vars(args),
            tags=[args.model, "baseline"],
        )

    probe = BaselineProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed, n_seeds=args.n_seeds)

    print(f"\n=== Baseline Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
