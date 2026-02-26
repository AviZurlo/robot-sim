"""Cosmos Policy adapter for the VLA probing suite.

Cosmos Policy is a video foundation model (Cosmos Predict2) fine-tuned for
visuomotor control. Unlike other VLAs in the suite, it operates on a latent
video diffusion paradigm: images are tokenized by a VAE, combined with
proprio/action/value latents into a temporal sequence, and denoised to
produce action chunks.

Architecture:
    Images (224x224) -> VAE encoder -> latent frames
    Proprio -> linear projection -> latent frame
    Language -> T5-XXL -> text conditioning embedding
    Latent sequence -> Cosmos Predict2 (diffusion transformer) -> denoised latents
    Action latent -> linear projection -> action chunk (16 steps × 7D)

Action format: 7-DoF end-effector deltas [x, y, z, roll, pitch, yaw, gripper]
Output is 16-step action chunk (configurable via chunk_size).

Requires CUDA — flash attention, triton, and megatron-core are hard deps.
"""

import os
import sys
from typing import Any

import numpy as np
import torch

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device


# Image size expected by Cosmos Policy
COSMOS_IMAGE_SIZE = 224


class CosmosPolicyAdapter(VLAAdapter):
    """Adapter for Cosmos Policy (nvidia/Cosmos-Policy-LIBERO-Predict2-2B).

    Cosmos Policy is unique in the probing suite:
        - Video diffusion model (not autoregressive or flow matching)
        - Outputs action chunks + future image predictions + value estimates
        - Uses T5-XXL text embeddings (not tokenized prompts)
        - Requires CUDA (flash attention, triton, megatron-core)
        - ~6.8GB VRAM for LIBERO inference

    For variance measurement, we use different seeds (0-9) for the
    diffusion sampling process, similar to X-VLA's flow matching seeds.
    """

    model_name = "cosmos_policy"

    # HuggingFace checkpoint
    CHECKPOINT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"

    def __init__(self) -> None:
        self.model = None
        self.cosmos_config = None
        self.cfg = None
        self.dataset_stats = None
        self.device = None
        self._cosmos_utils = None  # Lazy import of cosmos_policy utils

    @property
    def action_dim(self) -> int:
        return 7  # [x, y, z, roll, pitch, yaw, gripper]

    @property
    def chunk_size(self) -> int:
        return 16  # LIBERO default chunk size

    def _ensure_cosmos_importable(self) -> None:
        """Ensure cosmos_policy package is importable.

        The cosmos_policy repo must be cloned and available. We add it
        to sys.path if not already installed.
        """
        try:
            import cosmos_policy  # noqa: F401
        except ImportError:
            # Try common clone locations
            candidates = [
                "/workspace/cosmos-policy",
                "/tmp/cosmos-policy",
                os.path.expanduser("~/cosmos-policy"),
            ]
            for path in candidates:
                if os.path.isdir(os.path.join(path, "cosmos_policy")):
                    sys.path.insert(0, path)
                    print(f"Added {path} to sys.path for cosmos_policy")
                    return
            raise ImportError(
                "cosmos_policy not found. Clone it first:\n"
                "  git clone https://github.com/nvlabs/cosmos-policy.git /workspace/cosmos-policy\n"
                "  cd /workspace/cosmos-policy && pip install -e ."
            )

    def load_model(self, device: str = "cuda") -> None:
        """Load Cosmos Policy model and supporting resources.

        Args:
            device: Must be 'cuda' — Cosmos Policy requires CUDA.
        """
        if device == "mps":
            raise RuntimeError(
                "Cosmos Policy requires CUDA (flash-attn, triton, megatron-core). "
                "Cannot run on MPS. Use a CUDA GPU."
            )

        self._ensure_cosmos_importable()

        from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            load_dataset_stats,
            init_t5_text_embeddings_cache,
        )

        self.device = _get_device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"Cosmos Policy requires CUDA, got device: {self.device}"
            )

        print(f"Loading Cosmos Policy on {self.device}...")

        # Configure for LIBERO inference
        self.cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=self.CHECKPOINT,
            config_file="cosmos_policy/config/config.py",
            dataset_stats_path=f"{self.CHECKPOINT}/libero_dataset_statistics.json",
            t5_text_embeddings_path=f"{self.CHECKPOINT}/libero_t5_embeddings.pkl",
            use_wrist_image=True,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            chunk_size=16,
            num_open_loop_steps=16,
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            flip_images=True,  # LIBERO renders upside-down
            num_denoising_steps_action=5,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
        )

        # Load dataset stats for action/proprio normalization
        self.dataset_stats = load_dataset_stats(self.cfg.dataset_stats_path)

        # Initialize T5 text embeddings cache
        init_t5_text_embeddings_cache(self.cfg.t5_text_embeddings_path)

        # Load the model
        self.model, self.cosmos_config = get_model(self.cfg)

        # Store reference to cosmos_utils for get_action calls
        import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils
        self._cosmos_utils = cosmos_utils

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(
            f"Cosmos Policy loaded: {n_params:.1f}B params on {self.device}"
        )

    def _prepare_observation(self, inp: VLAInput) -> dict:
        """Convert VLAInput to Cosmos Policy observation dict.

        Cosmos Policy expects:
            - primary_image: third-person camera (H, W, 3) uint8
            - wrist_image: wrist camera (H, W, 3) uint8
            - proprio: proprioceptive state vector
        """
        # Primary = first image (agentview / third-person)
        primary_image = inp.images[0]

        # Wrist = second image if available, else duplicate primary
        if len(inp.images) > 1:
            wrist_image = inp.images[1]
        else:
            wrist_image = primary_image.copy()

        # Ensure images are uint8 numpy arrays
        if primary_image.dtype != np.uint8:
            primary_image = (primary_image * 255).astype(np.uint8)
        if wrist_image.dtype != np.uint8:
            wrist_image = (wrist_image * 255).astype(np.uint8)

        return {
            "primary_image": primary_image,
            "wrist_image": wrist_image,
            "proprio": inp.proprio.astype(np.float64),
        }

    def predict_action(self, inp: VLAInput, seed: int = 0) -> VLAOutput:
        """Run inference and return action chunk prediction.

        Uses Cosmos Policy's diffusion sampling to generate a 16-step
        action chunk. The seed controls the diffusion noise for
        stochasticity measurement.
        """
        obs = self._prepare_observation(inp)

        action_return_dict = self._cosmos_utils.get_action(
            cfg=self.cfg,
            model=self.model,
            dataset_stats=self.dataset_stats,
            obs=obs,
            task_label_or_embedding=inp.prompt,
            seed=seed,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )

        # Extract actions — shape: list of (7,) arrays or (chunk_size, 7)
        actions = action_return_dict["actions"]
        if isinstance(actions, list):
            actions = np.array(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        return VLAOutput(
            actions=actions,
            raw_output={
                "seed": seed,
                "value_prediction": action_return_dict.get("value_prediction"),
                "future_image_predictions": action_return_dict.get("future_image_predictions"),
            },
        )

    def predict_action_multi_seed(
        self, inp: VLAInput, n_seeds: int = 10
    ) -> list[VLAOutput]:
        """Run inference with multiple seeds for diffusion stochasticity."""
        results = []
        for seed in range(n_seeds):
            results.append(self.predict_action(inp, seed=seed))
        return results

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Attempt attention extraction from Cosmos Policy.

        Cosmos Policy uses a diffusion transformer architecture where
        attention maps are computed during the denoising process. Direct
        extraction is complex due to the multi-step denoising. We return
        empty attention maps and note this as a known limitation.

        The model's understanding of the scene is better probed through
        the other 7 probes (direction, spatial, etc.) rather than raw
        attention maps.
        """
        # Cosmos Policy's diffusion transformer doesn't expose attention
        # in the same way as encoder-decoder or autoregressive models.
        # The denoising process runs multiple steps, each with different
        # attention patterns. Meaningful extraction would require
        # aggregating across denoising steps.
        #
        # For now, return zeros — the attention probe will report IoU=0,
        # which is the honest result (same approach as π0).
        print(
            "WARNING: Cosmos Policy attention extraction not implemented. "
            "Diffusion transformer attention requires multi-step aggregation."
        )
        dummy_size = 16  # Small placeholder grid
        return {
            "spatial_attention": np.zeros((256, 256)),
            "raw_attention": np.zeros((dummy_size, dummy_size)),
            "patch_attention": np.zeros((dummy_size, dummy_size)),
            "n_image_tokens": 0,
            "patch_grid_size": dummy_size,
        }

    def reset(self) -> None:
        """No internal state to reset (each inference is independent)."""
