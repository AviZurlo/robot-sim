"""Probe 4: View ablation — remove primary/secondary views.

Tests which camera views the model relies on most by
zeroing out each view individually and measuring the impact.
"""

from typing import Any

import numpy as np

from vla_probing.adapter import VLAInput
from vla_probing.metrics import perturbation_sensitivity
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter, resolve_scene
from vla_probing.tracking import ExperimentTracker


class ViewAblationProbe(Probe):
    name = "view_ablation"
    description = "Remove primary/secondary camera views"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        prompt = "pick up the red block"

        # 1. Baseline (both views)
        self.scene.reset()
        baseline_actions, views = self._predict(prompt)
        baseline_2d = np.atleast_2d(baseline_actions).reshape(
            -1, baseline_actions.shape[-1]
        )

        ee_state = self.scene.get_ee_state()

        # 2. Remove primary view (zero it out)
        blank = np.zeros_like(views["image"])
        inp_no_primary = VLAInput(
            images=[blank, views["image2"]],
            prompt=prompt,
            proprio=ee_state,
        )
        self.adapter.reset()
        no_primary_output = self.adapter.predict_action(inp_no_primary)
        no_primary_2d = np.atleast_2d(no_primary_output.actions).reshape(
            -1, no_primary_output.actions.shape[-1]
        )

        # 3. Remove secondary view (zero it out)
        inp_no_secondary = VLAInput(
            images=[views["image"], blank],
            prompt=prompt,
            proprio=ee_state,
        )
        self.adapter.reset()
        no_secondary_output = self.adapter.predict_action(inp_no_secondary)
        no_secondary_2d = np.atleast_2d(no_secondary_output.actions).reshape(
            -1, no_secondary_output.actions.shape[-1]
        )

        # 4. Remove both views
        inp_no_vision = VLAInput(
            images=[blank, blank],
            prompt=prompt,
            proprio=ee_state,
        )
        self.adapter.reset()
        no_vision_output = self.adapter.predict_action(inp_no_vision)
        no_vision_2d = np.atleast_2d(no_vision_output.actions).reshape(
            -1, no_vision_output.actions.shape[-1]
        )

        extra_metrics = {
            "primary_ablation_sensitivity": perturbation_sensitivity(
                baseline_2d, no_primary_2d
            ),
            "secondary_ablation_sensitivity": perturbation_sensitivity(
                baseline_2d, no_secondary_2d
            ),
            "full_vision_ablation_sensitivity": perturbation_sensitivity(
                baseline_2d, no_vision_2d
            ),
        }

        return self._make_result(
            variant="zero_out_views",
            actions=baseline_2d,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "primary_view": views["image"],
                "secondary_view": views["image2"],
            },
        )


def main() -> None:
    parser = common_args("Probe 4: View ablation")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = resolve_scene(args)
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"view_ablation_{args.model}",
            config=vars(args),
            tags=[args.model, "view_ablation"],
        )

    probe = ViewAblationProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== View Ablation Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
