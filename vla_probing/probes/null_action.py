"""Probe 6: Null action prompts — 'don't move' should produce minimal motion.

Tests whether the model understands negative/null instructions
by measuring action magnitude for prompts like "don't move" or "stay still".
"""

from typing import Any

import numpy as np

from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter, resolve_scene
from vla_probing.tracking import ExperimentTracker

NULL_PROMPTS = [
    "don't move",
    "stay still",
    "do nothing",
    "remain in place",
    "hold position",
]

# Also test spatial primitives that Avik found produce grasping-like motions
SPATIAL_PRIMITIVES = [
    "move up",
    "move down",
    "move forward",
    "move backward",
]


class NullActionProbe(Probe):
    name = "null_action"
    description = "Test null action compliance with 'don't move' prompts"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        self.scene.reset()

        ee_state = self.scene.get_ee_state()
        ee_pos = ee_state[:3]

        # Run null prompts
        null_results = {}
        for prompt in NULL_PROMPTS:
            torch.manual_seed(seed)
            actions, _ = self._predict(prompt)
            actions_2d = np.atleast_2d(actions).reshape(-1, actions.shape[-1])
            traj_xyz = actions_2d[:, :3]

            # Measure displacement from current EE position
            displacement = np.linalg.norm(traj_xyz - ee_pos, axis=-1)
            null_results[prompt] = {
                "mean_displacement": float(np.mean(displacement)),
                "max_displacement": float(np.max(displacement)),
                "actions": actions_2d,
            }

        # Run spatial primitives for comparison
        spatial_results = {}
        for prompt in SPATIAL_PRIMITIVES:
            torch.manual_seed(seed)
            actions, _ = self._predict(prompt)
            actions_2d = np.atleast_2d(actions).reshape(-1, actions.shape[-1])
            traj_xyz = actions_2d[:, :3]
            displacement = np.linalg.norm(traj_xyz - ee_pos, axis=-1)
            spatial_results[prompt] = {
                "mean_displacement": float(np.mean(displacement)),
                "max_displacement": float(np.max(displacement)),
            }

        # Also get baseline "pick up" for reference
        torch.manual_seed(seed)
        baseline_actions, _ = self._predict("pick up the red block")
        baseline_2d = np.atleast_2d(baseline_actions).reshape(
            -1, baseline_actions.shape[-1]
        )
        baseline_disp = float(
            np.mean(np.linalg.norm(baseline_2d[:, :3] - ee_pos, axis=-1))
        )

        # Metrics
        null_displacements = [v["mean_displacement"] for v in null_results.values()]
        extra_metrics = {
            "mean_null_displacement": float(np.mean(null_displacements)),
            "max_null_displacement": float(np.max(null_displacements)),
            "baseline_pick_displacement": baseline_disp,
            "null_vs_baseline_ratio": float(np.mean(null_displacements))
            / max(baseline_disp, 1e-8),
        }

        # Per-prompt metrics
        for prompt, res in null_results.items():
            key = prompt.replace(" ", "_").replace("'", "")
            extra_metrics[f"null_{key}_displacement"] = res["mean_displacement"]

        for prompt, res in spatial_results.items():
            key = prompt.replace(" ", "_")
            extra_metrics[f"spatial_{key}_displacement"] = res["mean_displacement"]

        # Use the first null prompt's actions as the primary result
        first_null_actions = null_results[NULL_PROMPTS[0]]["actions"]

        return self._make_result(
            variant="null_prompts",
            actions=first_null_actions,
            seed=seed,
            extra_metrics=extra_metrics,
        )


def main() -> None:
    parser = common_args("Probe 6: Null action")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = resolve_scene(args)
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"null_action_{args.model}",
            config=vars(args),
            tags=[args.model, "null_action"],
        )

    probe = NullActionProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== Null Action Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
