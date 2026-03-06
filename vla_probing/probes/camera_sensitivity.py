"""Probe 3: Camera sensitivity — mirror/rotate camera view.

Tests whether spatial understanding is tied to camera pose
by mirroring the primary camera and observing trajectory changes.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import perturbation_sensitivity
from vla_probing.tracking import ProbeResult

from .base import Probe, common_args, make_adapter, resolve_scene
from vla_probing.tracking import ExperimentTracker


class CameraSensitivityProbe(Probe):
    name = "camera_sensitivity"
    description = "Mirror/rotate camera to test spatial understanding"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        prompt = "pick up the red block"

        # 1. Baseline prediction
        self.scene.reset()
        baseline_actions, baseline_views = self._predict(prompt, seed=seed)
        baseline_2d = np.atleast_2d(baseline_actions).reshape(
            -1, baseline_actions.shape[-1]
        )
        baseline_xyz = baseline_2d[:, :3]

        # 2. Mirrored camera prediction — same seed so randomness doesn't inflate sensitivity
        self.scene.reset()
        self.scene.mirror_camera()
        mirrored_actions, mirrored_views = self._predict(prompt, seed=seed)
        mirrored_2d = np.atleast_2d(mirrored_actions).reshape(
            -1, mirrored_actions.shape[-1]
        )
        mirrored_xyz = mirrored_2d[:, :3]

        # Reset camera
        self.scene.reset_camera()

        # 3. Software-mirrored image (flip image but keep camera normal)
        self.scene.reset()
        views = self.scene.render_all_views()
        flipped_primary = np.fliplr(views["image"]).copy()
        from vla_probing.adapter import VLAInput

        inp = VLAInput(
            images=[flipped_primary, views["image2"]],
            prompt=prompt,
            proprio=self.scene.get_ee_state(),
        )
        self.adapter.reset()
        self.adapter.seed_for_inference(seed)
        flip_output = self.adapter.predict_action(inp)
        flip_2d = np.atleast_2d(flip_output.actions).reshape(
            -1, flip_output.actions.shape[-1]
        )
        flip_xyz = flip_2d[:, :3]

        extra_metrics = {
            "mirror_camera_sensitivity": perturbation_sensitivity(
                baseline_2d, mirrored_2d
            ),
            "flip_image_sensitivity": perturbation_sensitivity(
                baseline_2d, flip_2d
            ),
        }

        return self._make_result(
            variant="mirror_primary_camera",
            actions=mirrored_2d,
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "baseline_view": baseline_views["image"],
                "mirrored_view": mirrored_views["image"],
                "flipped_view": flipped_primary,
                "baseline_xyz": baseline_xyz,
                "mirrored_xyz": mirrored_xyz,
                "flipped_xyz": flip_xyz,
            },
        )


def main() -> None:
    parser = common_args("Probe 3: Camera sensitivity")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = resolve_scene(args)
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"camera_sensitivity_{args.model}",
            config=vars(args),
            tags=[args.model, "camera_sensitivity"],
        )

    probe = CameraSensitivityProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== Camera Sensitivity Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
