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

Action format: embodiment-specific, determined by EmbodimentTag.
Output is multi-step action chunk (16 steps default).

Requires CUDA — uses flash attention in the DiT and VLM backbone.

Zero-shot probing approach: We use the base GR00T-N1.6-3B checkpoint
(not finetuned on LIBERO) with ROBOCASA_PANDA_OMRON embodiment tag,
which is the Panda variant present in the pretraining data. This gives
us a true zero-shot reading on our standard Franka scene.
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

    We use the base checkpoint with ROBOCASA_PANDA_OMRON tag (Panda in
    pretrain data) for zero-shot evaluation on our standard Franka scene.
    No LIBERO-specific optimization — consistent with Avik's methodology.

    For variance measurement, we use different random seeds for the
    flow matching sampling process (similar to X-VLA and Cosmos Policy).
    """

    model_name = "groot"

    # Base checkpoint — zero-shot (no LIBERO finetuning)
    CHECKPOINT = "nvidia/GR00T-N1.6-3B"

    # GR00T uses joint-space proprio via embodiment-specific MLPs
    use_joint_state = True

    def __init__(self) -> None:
        self.policy = None
        self.device = None
        self._embodiment_tag = None
        self._modality_config = None
        self._video_keys = None
        self._state_keys = None
        self._action_keys = None

    @property
    def action_dim(self) -> int:
        return 7  # Panda: 7-DoF arm actions

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

        Uses base checkpoint with ROBOCASA_PANDA_OMRON tag for zero-shot
        probing. Falls back through candidate tags if one doesn't work.

        Args:
            device: Must be 'cuda' — GR00T requires flash attention.
        """
        if device == "mps":
            raise RuntimeError(
                "GR00T N1.6 requires CUDA (flash attention). Cannot run on MPS."
            )

        self._ensure_groot_importable()

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.device = _get_device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"GR00T N1.6 requires CUDA, got device: {self.device}"
            )

        print(f"Loading GR00T N1.6 on {self.device}...")
        print(f"Checkpoint: {self.CHECKPOINT} (base, zero-shot)")

        # Try embodiment tags in preference order:
        # 1. ROBOCASA_PANDA_OMRON — Panda in pretrain data (best for zero-shot)
        # 2. LIBERO_PANDA — post-train registered (may work but not pretrained)
        # 3. NEW_EMBODIMENT — generic fallback
        tag_candidates = [
            ("ROBOCASA_PANDA_OMRON", EmbodimentTag.ROBOCASA_PANDA_OMRON),
            ("LIBERO_PANDA", EmbodimentTag.LIBERO_PANDA),
            ("NEW_EMBODIMENT", EmbodimentTag.NEW_EMBODIMENT),
        ]

        for tag_name, tag in tag_candidates:
            try:
                print(f"Trying embodiment tag: {tag_name}...")
                self._embodiment_tag = tag
                self.policy = Gr00tPolicy(
                    model_path=__import__("huggingface_hub").snapshot_download(self.CHECKPOINT, local_files_only=True),
                    embodiment_tag=tag,
                    device=str(self.device),
                    strict=False,  # Relaxed validation for probing
                )
                print(f"SUCCESS: Loaded with {tag_name}")
                break
            except Exception as e:
                print(f"  {tag_name} failed: {e}")
                self.policy = None
                continue

        if self.policy is None:
            raise RuntimeError(
                "Failed to load GR00T N1.6 with any embodiment tag. "
                "Check that Isaac-GR00T is installed and checkpoint is downloaded."
            )

        # Query modality config to understand expected I/O
        self._modality_config = self.policy.get_modality_config()
        self._video_keys = self._modality_config["video"].modality_keys
        self._state_keys = self._modality_config["state"].modality_keys
        self._action_keys = self._modality_config["action"].modality_keys

        print(f"Expected video keys: {self._video_keys}")
        print(f"Expected state keys: {self._state_keys}")
        print(f"Expected action keys: {self._action_keys}")

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

        Camera and state key names are determined by the embodiment tag's
        modality config. We map our scene's outputs to whatever keys
        the model expects.
        """
        # Primary image — add batch and temporal dims: (1, 1, H, W, 3)
        primary = inp.images[0]
        if primary.dtype != np.uint8:
            primary = (primary * 255).astype(np.uint8)
        primary = primary[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 3)

        # Map our cameras to expected video keys
        video = {}
        if self._video_keys:
            # First video key gets primary camera
            video[self._video_keys[0]] = primary
            # Second video key gets wrist/secondary camera if available
            if len(self._video_keys) > 1 and len(inp.images) > 1:
                wrist = inp.images[1]
                if wrist.dtype != np.uint8:
                    wrist = (wrist * 255).astype(np.uint8)
                wrist = wrist[np.newaxis, np.newaxis, ...]
                video[self._video_keys[1]] = wrist
        else:
            # Fallback to standard names
            video["agentview"] = primary
            if len(inp.images) > 1:
                wrist = inp.images[1]
                if wrist.dtype != np.uint8:
                    wrist = (wrist * 255).astype(np.uint8)
                video["robot0_eye_in_hand"] = wrist[np.newaxis, np.newaxis, ...]

        # Proprio state — (1, 1, D) float32
        proprio = inp.proprio.astype(np.float32)
        state = {}
        if self._state_keys:
            # Use the first state key
            state[self._state_keys[0]] = proprio[np.newaxis, np.newaxis, ...]
        else:
            state["joint_position"] = proprio[np.newaxis, np.newaxis, ...]

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

        # Extract actions — use known action keys or find first array
        actions = None
        if self._action_keys:
            for key in self._action_keys:
                if key in action_dict and isinstance(action_dict[key], np.ndarray):
                    actions = action_dict[key]
                    break

        if actions is None:
            # Fallback: find first numpy array in output
            for key, val in action_dict.items():
                if isinstance(val, np.ndarray):
                    actions = val
                    break

        if actions is None:
            raise RuntimeError(
                f"No action array in GR00T output: {action_dict.keys()}"
            )

        # Shape: (B, T, D) -> (T, D)
        if actions.ndim == 3:
            actions = actions[0]  # Remove batch dim
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        return VLAOutput(
            actions=actions,
            raw_output={
                "seed": seed,
                "info": info,
                "embodiment_tag": str(self._embodiment_tag),
                "action_keys": self._action_keys,
            },
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
        return {
            "spatial_attention": np.zeros((256, 256)),
            "n_image_tokens": 0,
        }

    def reset(self) -> None:
        """Reset policy state between episodes."""
        if self.policy is not None:
            try:
                self.policy.reset()
            except Exception:
                pass  # Policy may be stateless
