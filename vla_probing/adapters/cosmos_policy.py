"""Cosmos Policy adapter for the VLA probing suite.

Cosmos Policy (NVIDIA) fine-tunes a Cosmos Predict2 video model for robot
control. Unlike standard VLAs that use vision-language backbones, Cosmos
Policy leverages a video generation model to understand physical dynamics.

Architecture:
    Images (224x224) -> Cosmos Predict2 (2B video model)
    Language -> T5 text embeddings
    -> Diffusion-based denoising -> action chunks + future images + value

Action format: 7-DoF for LIBERO (same as π0), 16-step chunks
Also predicts future images and value estimates (unique to this model).

REQUIRES CUDA — this adapter cannot run on MPS or CPU.
Use scripts/run_cosmos_cloud.sh on a cloud GPU instance.

Checkpoint: nvidia/Cosmos-Policy-LIBERO-Predict2-2B
"""

from typing import Any

import numpy as np
import torch

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput


# Standard image size for Cosmos Policy
COSMOS_IMAGE_SIZE = 224


class CosmosPolicyAdapter(VLAAdapter):
    """Adapter for Cosmos Policy (nvidia/Cosmos-Policy-LIBERO-Predict2-2B).

    This model is architecturally distinct from all other models in the suite:
    it uses a video generation backbone (Cosmos Predict2) rather than a VLM.
    Actions are generated via diffusion-based denoising in latent video space.

    Unique capabilities:
        - Predicts future images (what the scene should look like)
        - Predicts value estimates (expected cumulative reward)
        - Uses flow matching in video latent space for action generation

    Variance: Uses seed-based sampling like flow matching models.
    """

    model_name = "cosmos_policy"

    def __init__(self) -> None:
        self.model = None
        self.config = None
        self.cosmos_config = None
        self.dataset_stats = None
        self.device = None
        self._seed = 0

    @property
    def action_dim(self) -> int:
        return 7  # LIBERO: [x, y, z, roll, pitch, yaw, gripper]

    @property
    def chunk_size(self) -> int:
        return 16  # Cosmos Policy predicts 16-step action chunks

    def load_model(self, device: str = "cuda") -> None:
        """Load Cosmos Policy LIBERO checkpoint.

        IMPORTANT: This model requires CUDA. It will not work on MPS or CPU
        due to deep CUDA dependencies in the Cosmos Predict2 backbone.
        """
        # Import cosmos_policy modules (must be on PYTHONPATH)
        from cosmos_policy.experiments.robot.libero.run_libero_eval import (
            PolicyEvalConfig,
        )
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        if device != "cuda" and not device.startswith("cuda:"):
            print(
                f"WARNING: Cosmos Policy requires CUDA. Got device={device}. "
                "Attempting CUDA anyway..."
            )

        self.device = torch.device("cuda:0")

        # Configure for LIBERO evaluation
        self.config = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
            config_file="cosmos_policy/config/config.py",
            dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
            t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
            use_wrist_image=True,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            chunk_size=16,
            num_open_loop_steps=16,
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            flip_images=True,  # LIBERO images render upside-down
            num_denoising_steps_action=5,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
        )

        # Load dataset stats for action/proprio scaling
        self.dataset_stats = load_dataset_stats(self.config.dataset_stats_path)

        # Initialize T5 text embeddings cache
        init_t5_text_embeddings_cache(self.config.t5_text_embeddings_path)

        # Load model
        print("Loading Cosmos Policy LIBERO checkpoint...")
        self.model, self.cosmos_config = get_model(self.config)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"Cosmos Policy loaded: {n_params:.1f}B params on {self.device}")

    def _prepare_observation(self, inp: VLAInput) -> dict[str, Any]:
        """Convert VLAInput to Cosmos Policy observation format.

        Cosmos Policy LIBERO expects:
            - primary_image: third-person view (H, W, 3) uint8
            - wrist_image: wrist camera view (H, W, 3) uint8
            - proprio: proprioceptive state (8D for LIBERO)
        """
        import cv2

        # Map our scene cameras to Cosmos Policy's expected format
        # image = primary (third-person), image2 = wrist camera
        primary = inp.images[0]  # (H, W, 3) uint8
        wrist = inp.images[1] if len(inp.images) > 1 else inp.images[0]

        # Resize to 224x224 (Cosmos Policy's expected size)
        if primary.shape[:2] != (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE):
            primary = cv2.resize(
                primary,
                (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        if wrist.shape[:2] != (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE):
            wrist = cv2.resize(
                wrist,
                (COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        obs = {
            "primary_image": primary,
            "wrist_image": wrist,
        }

        # Add proprioception if available
        if inp.proprio is not None:
            obs["proprio"] = inp.proprio[:8].astype(np.float32)

        return obs

    def predict_action(self, inp: VLAInput) -> VLAOutput:
        """Run inference and return action chunk.

        Uses Cosmos Policy's get_action() which handles:
        1. T5 text embedding of the language instruction
        2. Image preprocessing and latent encoding
        3. Diffusion-based denoising to generate actions
        4. Action unnormalization to original dataset scale
        """
        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        obs = self._prepare_observation(inp)

        # get_action returns a dict with 'actions', 'future_image_predictions', 'value_prediction'
        result = get_action(
            self.config,
            self.model,
            self.dataset_stats,
            obs,
            inp.prompt,  # T5 embedding computed on-the-fly if not cached
            seed=self._seed,
            num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=True,
        )

        # Extract action chunk: list of 16 action arrays -> (16, 7) numpy
        actions = result["actions"]
        if isinstance(actions, list):
            actions = np.array(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # Store extra outputs for potential analysis
        raw_output = {
            "value_prediction": (
                result.get("value_prediction", None)
            ),
        }

        # Save future image predictions if available
        future_imgs = result.get("future_image_predictions", None)
        if future_imgs is not None:
            raw_output["future_image_predictions"] = future_imgs

        return VLAOutput(
            actions=actions,
            raw_output=raw_output,
        )

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Extract attention from Cosmos Policy.

        Cosmos Policy uses a video diffusion transformer, not a standard
        VLM with text-to-vision attention. Attention extraction would
        require hooking into the DiT's cross-attention layers.

        For now, returns empty dict — attention probing for video models
        is an open research question.
        """
        print(
            "WARNING: Attention extraction not yet implemented for Cosmos Policy. "
            "The DiT architecture uses different attention patterns than VLMs."
        )
        return {
            "spatial_attention": np.zeros((256, 256)),
            "n_image_tokens": 0,
        }

    def reset(self) -> None:
        """Reset seed for next prediction."""
        self._seed = (self._seed + 1) % 256
