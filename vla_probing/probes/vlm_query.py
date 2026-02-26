"""Probe 0: VLM scene querying — test OpenVLA's ability to describe scenes.

This probe is UNIQUE to OpenVLA because it's the only model in the suite
where actions are text tokens from the same output head as language.
We can prompt the model for free-form text descriptions instead of actions.

Tests:
- Scene description ("Describe what you see on the table")
- Object identification ("What color is the block on the left?")
- Spatial reasoning ("Which block is closer to the robot?")
- Counting ("How many blocks are on the table?")

This probe is infeasible for X-VLA and pi0 (both have separate
action heads that can't generate language).
"""

from typing import Any

import numpy as np

from vla_probing.adapter import VLAInput
from vla_probing.tracking import ExperimentTracker, ProbeResult

from .base import Probe, common_args, make_adapter, resolve_scene


# VLM query prompts organized by category
VLM_QUERIES = {
    "scene_description": [
        "Describe what you see on the table",
        "What objects are visible in this image?",
    ],
    "object_identification": [
        "What color is the block on the left?",
        "What color is the block on the right?",
    ],
    "spatial_reasoning": [
        "Which block is closer to the robot?",
        "Describe the positions of the blocks relative to each other",
    ],
    "counting": [
        "How many blocks are on the table?",
        "How many objects can you see?",
    ],
}


class VLMQueryProbe(Probe):
    """Probe 0: VLM scene querying (OpenVLA-only).

    Tests whether OpenVLA can describe scenes in addition to predicting
    actions, exploiting the shared text-token output head.
    """

    name = "vlm_query"
    description = "VLM scene querying (OpenVLA-only)"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        self.scene.reset()

        # Check that adapter supports VLM querying
        if not hasattr(self.adapter, "query_vlm"):
            return self._make_result(
                variant="unsupported",
                actions=np.zeros((1, self.adapter.action_dim)),
                seed=seed,
                extra_metrics={"vlm_supported": 0.0},
                artifacts={"note": "Model does not support VLM querying"},
            )

        views = self.scene.render_all_views()
        images = [views["image"], views["image2"]]

        responses: dict[str, dict[str, str]] = {}
        response_lengths: list[int] = []
        non_empty_count = 0
        total_count = 0

        for category, prompts in VLM_QUERIES.items():
            responses[category] = {}
            for prompt in prompts:
                total_count += 1
                print(f"  Query: {prompt}")
                response = self.adapter.query_vlm(images, prompt)
                responses[category][prompt] = response
                print(f"  Response: {response!r}")

                # Track response quality metrics
                response_lengths.append(len(response))
                if len(response.strip()) > 0:
                    non_empty_count += 1

        # Compute metrics
        extra_metrics = {
            "vlm_supported": 1.0,
            "non_empty_response_rate": non_empty_count / max(total_count, 1),
            "mean_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
            "total_queries": float(total_count),
            "non_empty_responses": float(non_empty_count),
        }

        # Check for action-token contamination (responses that look like action bins)
        action_like = 0
        for cat_responses in responses.values():
            for resp in cat_responses.values():
                # Action tokens are typically short numeric patterns
                tokens = resp.split()
                if all(t.replace(".", "").replace("-", "").isdigit() for t in tokens[:3]):
                    action_like += 1
        extra_metrics["action_token_contamination_rate"] = action_like / max(total_count, 1)

        return self._make_result(
            variant="vlm_scene_queries",
            actions=np.zeros((1, self.adapter.action_dim)),
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "primary_view": views["image"],
                "responses": responses,
            },
        )


def main() -> None:
    parser = common_args("Probe 0: VLM scene querying (OpenVLA-only)")
    args = parser.parse_args()

    if args.model != "openvla":
        print(f"VLM querying probe only works with OpenVLA, not {args.model}")
        return

    adapter = make_adapter(args.model, args.device)
    scene = resolve_scene(args)
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"vlm_query_{args.model}",
            config=vars(args),
            tags=[args.model, "vlm_query"],
        )

    probe = VLMQueryProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== VLM Query Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
