"""GR00T N1.6 adapter for the VLA probing suite.

GR00T N1.6 is NVIDIA's cross-embodiment VLA model combining a
vision-language foundation model (Cosmos-Reason-2B) with a diffusion
transformer (DiT) head that denoises continuous actions via flow matching.

Architecture:
    Images -> SigLIP2 vision transformer -> patch embeddings
    Language -> T5 text encoder -> text embeddings
    Proprio -> embodiment-specific MLP -> state embeddings
    [vision + language] -> cross-attention conditioning
    [state + noise] -> DiT (32 layers) -> denoised action chunk

Action format: embodiment-specific (7D for LIBERO: EEF deltas + gripper)
Output is multi-step action chunk (16 steps default).

Requires CUDA — uses flash attention in the DiT and VLM backbone.
"""

import os
import sys
from typing import Any

import numpy as np
import torch

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device


class GR00TAdapter(VLAAdapter):
    """Adapter for GR00T N1.6 (nvidia/GR00T-N1.6-3B).

    GR00T N1.6 is unique in the probing suite:
        - Cross-embodiment VLA (trained on humanoid + bimanual + manipulation)
        - Uses flow matching DiT for action denoising (not autoregressive)
        - Vision backbone: SigLIP2 (not DINOv2 or Prismatic)
        - Language: T5 text encoder (not tokenized into LLM)
        - Embodiment-specific proprio/action MLPs
        - 3B parameters total

    For LIBERO evaluation, uses the finetuned checkpoint with
    EmbodimentTag for LIBERO's Franka setup.

    For variance measurement, we use different random seeds for the
    flow matching sampling process (similar to X-VLA and Cosmos Policy).
    """

    model_name = "groot"

    # HuggingFace checkpoint — LIBERO finetuned
    CHECKPOINT = "nvidia/GR00T-N1.6-3B"
    # Fall back to LIBERO-spatial finetuned if available
    CHECKPOINT_LIBERO = "nvidia/GR00T-N1.6-libero-spatial"

    def __init__(self) -> None:
        self.policy = None
        self.device = None
        self._embodiment_tag = None

    @property
    def action_dim(self) -> int:
        return 7  # LIBERO: [dx, dy, dz, droll, dpitch, dyaw, gripper]

    @property
    def chunk_size(self) -> int:
        return 16  # Default action horizon

    def _ensure_groot_importable(self) -> None:
        """Ensure gr00t package is importable."""
        try:
            import gr00t  # noqa: F401
        except ImportError:
            candidates = [
                "/workspace/Isaac-GR00T",
                "/tmp/Isaac-GR00T",
                os.path.expanduser("~/Isaac-GR00T"),
            ]
            for path in candidates:
                if os.path.isdir(os.path.join(path, "gr00t")):
                    sys.path.insert(0, path)
                    print(f"Added {path} to sys.path for gr00t")
                    return
            raise ImportError(
                "Isaac-GR00T not found. Clone it first:\n"
                "  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git /workspace/Isaac-GR00T\n"
                "  cd /workspace/Isaac-GR00T && pip install -e ."
            )

    def load_model(self, device: str = "cuda") -> None:
        """Load GR00T N1.6 policy.

        Args:
            device: Must be 'cuda' — GR00T requires flash attention.
        """
        if device == "mps":
            raise RuntimeError(
                "GR00T N1.6 requires CUDA (flash attention). Cannot run on MPS."
            )

        self._ensure_groot_importable()

        from gr00t.policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.device = _get_device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"GR00T N1.6 requires CUDA, got device: {self.device}"
            )

        print(f"Loading GR00T N1.6 on {self.device}...")

        # Try LIBERO-specific checkpoint first, fall back to base
        checkpoint = self.CHECKPOINT
        try:
            from huggingface_hub import model_info
            model_info(self.CHECKPOINT_LIBERO)
            checkpoint = self.CHECKPOINT_LIBERO
            print(f"Using LIBERO-finetuned checkpoint: {checkpoint}")
        except Exception:
            print(f"Using base checkpoint: {checkpoint}")

        # Determine LIBERO embodiment tag
        # GR00T uses EmbodimentTag to configure proprio/action dims
        self._embodiment_tag = EmbodimentTag.LIBERO

        self.policy = Gr00tPolicy(
            model_path=checkpoint,
            embodiment_tag=self._embodiment_tag,
            device=str(self.device),
            strict=False,  # Relaxed validation for probing
        )

        n_params = sum(
            p.numel() for p in self.policy.model.parameters()
        ) / 1e9
        print(f"GR00T N1.6 loaded: {n_params:.1f}B params on {self.device}")

    def _prepare_observation(self, inp: VLAInput) -> dict:
        """Convert VLAInput to GR00T observation dict.

        GR00T expects:
            video: {camera_name: (B, T, H, W, 3) uint8}
            state: {state_name: (B, T, D) float32}
            language: {task: [[str]]}
        """
        # Primary image — add batch and temporal dims: (1, 1, H, W, 3)
        primary = inp.images[0]
        if primary.dtype != np.uint8:
            primary = (primary * 255).astype(np.uint8)
        primary = primary[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 3)

        video = {"agentview": primary}

        # Add wrist camera if available
        if len(inp.images) > 1:
            wrist = inp.images[1]
            if wrist.dtype != np.uint8:
                wrist = (wrist * 255).astype(np.uint8)
            wrist = wrist[np.newaxis, np.newaxis, ...]
            video["robot0_eye_in_hand"] = wrist

        # Proprio state — (1, 1, D) float32
        state = {
            "joint_position": inp.proprio.astype(np.float32)[np.newaxis, np.newaxis, ...]
        }

        # Language instruction
        language = {"task": [[inp.prompt]]}

        return {
            "video": video,
            "state": state,
            "language": language,
        }

    def predict_action(self, inp: VLAInput, seed: int = 0) -> VLAOutput:
        """Run inference and return action chunk prediction.

        Uses flow matching sampling with configurable seed for
        stochasticity measurement.
        """
        torch.manual_seed(seed)
        obs = self._prepare_observation(inp)

        action_dict, info = self.policy.get_action(obs)

        # Extract actions — find the action key
        actions = None
        for key, val in action_dict.items():
            if isinstance(val, np.ndarray):
                actions = val
                break

        if actions is None:
            raise RuntimeError(f"No action array in GR00T output: {action_dict.keys()}")

        # Shape: (B, T, D) -> (T, D)
        if actions.ndim == 3:
            actions = actions[0]  # Remove batch dim
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        return VLAOutput(
            actions=actions,
            raw_output={"seed": seed, "info": info},
        )

    def predict_action_multi_seed(
        self, inp: VLAInput, n_seeds: int = 10
    ) -> list[VLAOutput]:
        """Run inference with multiple seeds for flow matching stochasticity."""
        results = []
        for seed in range(n_seeds):
            results.append(self.predict_action(inp, seed=seed))
        return results

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Attempt attention extraction from GR00T N1.6.

        GR00T uses cross-attention between the DiT and VLM features.
        Direct extraction would require hooking into the DiT's
        cross-attention layers. For now, return zeros.
        """
        print(
            "WARNING: GR00T attention extraction not implemented. "
            "DiT cross-attention extraction requires custom hooks."
        )
        dummy_size = 16
        return {
            "spatial_attention": np.zeros((256, 256)),
            "raw_attention": np.zeros((dummy_size, dummy_size)),
            "patch_attention": np.zeros((dummy_size, dummy_size)),
            "n_image_tokens": 0,
            "patch_grid_size": dummy_size,
        }

    def reset(self) -> None:
        """No internal state to reset."""
