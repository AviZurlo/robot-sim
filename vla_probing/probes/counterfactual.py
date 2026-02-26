"""Probe 5: Counterfactual prompts — synonym variations.

Tests whether the language encoder understands synonyms by using
different phrasings that should produce the same action trajectory.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import perturbation_sensitivity
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker

# Prompt groups: each group should produce similar actions
SYNONYM_GROUPS = {
    "red_block_synonyms": [
        "pick up the red block",
        "pick up the red cube",
        "grab the red block",
        "grasp the crimson block",
        "pick up the red object",
    ],
    "spatial_primitives": [
        "move up",
        "move down",
        "move forward",
        "move left",
        "move right",
    ],
    "visually_grounded": [
        "move toward the red block",
        "move toward the blue block",
        "move toward the base",
    ],
}


class CounterfactualProbe(Probe):
    name = "counterfactual"
    description = "Test language understanding with synonym variations"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        group = kwargs.get("group", "red_block_synonyms")
        prompts = SYNONYM_GROUPS.get(group, SYNONYM_GROUPS["red_block_synonyms"])

        self.scene.reset()

        # Run inference for each prompt variant
        all_actions = {}
        for prompt in prompts:
            torch.manual_seed(seed)  # same seed for fair comparison
            actions, _ = self._predict(prompt)
            actions_2d = np.atleast_2d(actions).reshape(-1, actions.shape[-1])
            all_actions[prompt] = actions_2d

        # Compute pairwise sensitivity between prompt variants
        baseline_prompt = prompts[0]
        baseline = all_actions[baseline_prompt]

        pairwise_sensitivity = {}
        for prompt in prompts[1:]:
            key = f"vs_{prompt.replace(' ', '_')[:30]}"
            pairwise_sensitivity[key] = perturbation_sensitivity(
                baseline, all_actions[prompt]
            )

        # Overall consistency: mean sensitivity across all pairs
        sensitivities = list(pairwise_sensitivity.values())
        extra_metrics = {
            "mean_synonym_sensitivity": float(np.mean(sensitivities))
            if sensitivities
            else 0.0,
            "max_synonym_sensitivity": float(np.max(sensitivities))
            if sensitivities
            else 0.0,
            **pairwise_sensitivity,
        }

        return self._make_result(
            variant=f"group_{group}",
            actions=baseline,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "all_actions": {k: v.tolist() for k, v in all_actions.items()},
            },
        )


def main() -> None:
    parser = common_args("Probe 5: Counterfactual prompts")
    parser.add_argument(
        "--group",
        default="red_block_synonyms",
        choices=list(SYNONYM_GROUPS.keys()),
        help="Which prompt group to test",
    )
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"counterfactual_{args.model}_{args.group}",
            config=vars(args),
            tags=[args.model, "counterfactual", args.group],
        )

    probe = CounterfactualProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed, group=args.group)

    print(f"\n=== Counterfactual Probe Results ({args.model}, {args.group}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
