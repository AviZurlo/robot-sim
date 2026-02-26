"""Probe 7: Attention visualization — extract and overlay attention maps.

Uses the VLAAdapter's get_attention() method to extract attention from
the VLM encoder and overlay it on input images. Identifies what image
regions the model attends to for different prompts and scenes.
"""

from typing import Any

import numpy as np

from vla_probing.metrics import attention_iou
from vla_probing.tracking import ProbeResult, create_attention_overlay

from .base import Probe, common_args, make_adapter
from vla_probing.adapter import VLAInput
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker


class AttentionProbe(Probe):
    name = "attention"
    description = "Extract and visualize attention maps"

    def run(self, seed: int = 0, **kwargs: Any) -> ProbeResult:
        import torch

        torch.manual_seed(seed)
        self.scene.reset()

        prompts = [
            "pick up the red block",
            "pick up the blue block",
            "move to the left",
        ]

        views = self.scene.render_all_views()
        primary_img = views["image"]
        ee_state = self.scene.get_ee_state()

        attention_maps = {}
        overlays = {}
        iou_scores = {}

        for prompt in prompts:
            inp = VLAInput(
                images=[views["image"], views["image2"]],
                prompt=prompt,
                proprio=ee_state,
            )

            attn_result = self.adapter.get_attention(inp)
            spatial_attn = attn_result["spatial_attention"]
            attention_maps[prompt] = spatial_attn

            # Create overlay
            overlay = create_attention_overlay(primary_img, spatial_attn)
            overlays[prompt] = overlay

            # Compute IoU with approximate object region
            obj_mask = self._make_object_mask(prompt, primary_img)
            if obj_mask is not None:
                iou = attention_iou(spatial_attn, obj_mask)
                key = prompt.replace(" ", "_")[:30]
                iou_scores[f"iou_{key}"] = iou

            # Log to tracker
            self.tracker.log_attention_overlay(
                primary_img,
                spatial_attn,
                name=f"attention_{prompt.replace(' ', '_')[:20]}",
            )

        extra_metrics = {**iou_scores}
        if iou_scores:
            extra_metrics["mean_attention_iou"] = float(
                np.mean(list(iou_scores.values()))
            )

        # Use first prompt's attention as primary result
        return self._make_result(
            variant="multi_prompt_attention",
            actions=np.zeros((1, 10)),  # attention probe doesn't focus on actions
            seed=seed,
            extra_metrics=extra_metrics,
            artifacts={
                "primary_view": primary_img,
                **{f"overlay_{k.replace(' ', '_')[:20]}": v for k, v in overlays.items()},
                **{
                    f"attention_{k.replace(' ', '_')[:20]}": v
                    for k, v in attention_maps.items()
                },
            },
        )

    def _make_object_mask(
        self, prompt: str, image: np.ndarray
    ) -> np.ndarray | None:
        """Create a rough object mask based on color segmentation.

        This is approximate — uses simple color thresholding to find
        red/blue block regions in the image.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        if "red" in prompt:
            # Detect red pixels (R > 150, G < 100, B < 100)
            mask = (image[:, :, 0] > 150) & (image[:, :, 1] < 100) & (image[:, :, 2] < 100)
        elif "blue" in prompt:
            # Detect blue pixels (B > 150, R < 100, G < 100)
            mask = (image[:, :, 2] > 150) & (image[:, :, 0] < 100) & (image[:, :, 1] < 100)
        else:
            return None

        # Dilate mask slightly to account for attention resolution
        if mask.any():
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=5)

        return mask.astype(np.float32) if mask.any() else None


def main() -> None:
    parser = common_args("Probe 7: Attention visualization")
    args = parser.parse_args()

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"attention_{args.model}",
            config=vars(args),
            tags=[args.model, "attention"],
        )

    probe = AttentionProbe(adapter, scene, tracker)
    result = probe.run(seed=args.seed)

    print(f"\n=== Attention Probe Results ({args.model}) ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.6f}")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
