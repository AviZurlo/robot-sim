"""VLAAdapter interface and X-VLA implementation.

Defines a base adapter class that all VLA models implement,
plus the concrete XVLAAdapter for the X-VLA WidowX checkpoint.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class VLAInput:
    """Standardized input for VLA inference."""

    images: list[np.ndarray]  # List of RGB images, each (H, W, 3), uint8
    prompt: str  # Language instruction
    proprio: np.ndarray  # Proprioceptive state vector


@dataclass
class VLAOutput:
    """Standardized output from VLA inference."""

    actions: np.ndarray  # Predicted actions, shape (chunk_size, action_dim)
    raw_output: dict[str, Any] | None = None  # Model-specific raw outputs


class VLAAdapter(ABC):
    """Base adapter class for VLA models.

    All VLA models implement this interface so probes can be model-agnostic.
    """

    model_name: str

    @abstractmethod
    def load_model(self, device: str = "mps") -> None:
        """Load model weights and tokenizer onto the specified device."""

    @abstractmethod
    def predict_action(self, inp: VLAInput) -> VLAOutput:
        """Run single-step inference and return predicted actions."""

    @abstractmethod
    def get_attention(
        self, inp: VLAInput
    ) -> dict[str, np.ndarray]:
        """Extract attention maps from the model.

        Returns a dict with keys like 'vision_attention', 'cross_attention', etc.
        Each value is an ndarray of attention weights.
        """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space."""

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Number of action steps predicted per inference call."""

    def reset(self) -> None:
        """Reset any internal state (action queues, caches). Override if needed."""


def _get_device(preferred: str = "mps") -> torch.device:
    """Resolve device, falling back gracefully."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class XVLAAdapter(VLAAdapter):
    """Adapter for X-VLA (lerobot/xvla-widowx checkpoint).

    X-VLA uses Florence2 as VLM backbone with DaViT vision encoder,
    soft-prompted transformer, and flow matching action head.
    Outputs absolute EE targets in 6D rotation representation.

    Action format per timestep (10D):
        [x, y, z, rot6d(6), gripper]
    Output is 20D = two packed timesteps of 10D each.
    """

    model_name = "xvla"

    def __init__(self) -> None:
        self.policy = None
        self.tokenizer = None
        self.device = None

    @property
    def action_dim(self) -> int:
        return 20  # 2 timesteps × 10D per timestep

    @property
    def chunk_size(self) -> int:
        if self.policy is not None:
            return self.policy.config.chunk_size
        return 30

    def load_model(self, device: str = "mps") -> None:
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from transformers import AutoTokenizer

        self.device = _get_device(device)
        print(f"Loading X-VLA on {self.device}...")

        # from_pretrained auto-detects device (CUDA→MPS fallback built in)
        self.policy = XVLAPolicy.from_pretrained("lerobot/xvla-widowx")
        self.policy.eval()
        self.device = next(self.policy.parameters()).device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.tokenizer_name
        )
        print(
            f"X-VLA loaded: {sum(p.numel() for p in self.policy.parameters()) / 1e6:.1f}M params on {self.device}"
        )

    def _prepare_batch(self, inp: VLAInput) -> dict[str, torch.Tensor]:
        """Convert VLAInput to the batch dict XVLAPolicy expects."""
        # Images: (H, W, 3) uint8 -> (1, 3, H, W) float [0, 1]
        imgs = []
        for img in inp.images[:2]:  # X-VLA uses 2 camera views
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(t.unsqueeze(0).to(self.device))

        # Pad to 2 images if only 1 provided
        while len(imgs) < 2:
            imgs.append(torch.zeros_like(imgs[0]))

        # Tokenize language instruction
        tokens = self.tokenizer(
            inp.prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
        )

        # State: BridgeData format [x, y, z, roll, pitch, yaw, 0, gripper]
        state = torch.from_numpy(inp.proprio).float().unsqueeze(0).to(self.device)

        return {
            "observation.images.image": imgs[0],
            "observation.images.image2": imgs[1],
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].to(self.device),
        }

    def predict_action(self, inp: VLAInput) -> VLAOutput:
        batch = self._prepare_batch(inp)
        with torch.no_grad():
            self.policy.reset()
            # predict_action_chunk returns full chunk: (1, chunk_size, 20)
            actions = self.policy.predict_action_chunk(batch)

        # (1, chunk_size, 20) -> (chunk_size, 20)
        actions_np = actions[0].cpu().numpy()
        return VLAOutput(actions=actions_np)

    def predict_action_multi_seed(
        self, inp: VLAInput, n_seeds: int = 10
    ) -> list[VLAOutput]:
        """Run inference with multiple random seeds for flow matching stochasticity."""
        results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            results.append(self.predict_action(inp))
        return results

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Extract attention maps from X-VLA's VLM encoder.

        Accesses the Florence2 BART encoder with output_attentions=True
        to get spatial attention over DaViT image patches and language tokens.
        """
        batch = self._prepare_batch(inp)

        with torch.no_grad():
            # Stack both camera views: (2, 3, H, W)
            image_input = torch.cat(
                [batch["observation.images.image"], batch["observation.images.image2"]],
                dim=0,
            )
            vision_feats = self.policy.model.vlm._encode_image(image_input)

            # Get language embeddings
            lang_embeds = self.policy.model.vlm.get_input_embeddings()(
                batch["observation.language.tokens"]
            )

            # Merge primary camera features + language tokens
            merged_embeds, attn_mask = (
                self.policy.model.vlm._merge_input_ids_with_image_features(
                    vision_feats[:1], lang_embeds
                )
            )

            # Run encoder with attention output
            encoder = self.policy.model.vlm.language_model.model.encoder
            enc_out = encoder(
                attention_mask=attn_mask,
                inputs_embeds=merged_embeds,
                output_attentions=True,
            )

        # Extract last layer attention, average over heads
        # Shape: (1, n_heads, seq_len, seq_len) -> (seq_len, seq_len)
        last_attn = enc_out.attentions[-1][0].mean(dim=0).cpu().numpy()

        # DaViT produces spatial patches — determine grid size from feature count
        n_image_tokens = vision_feats.shape[1]
        patch_grid_size = int(np.sqrt(n_image_tokens))
        n_spatial = patch_grid_size * patch_grid_size

        # Average attention TO image patches across all query positions
        img_attn = last_attn[:, :n_spatial].mean(axis=0)
        spatial_attn = img_attn.reshape(patch_grid_size, patch_grid_size)

        # Upscale to image resolution (256x256)
        scale = 256 // patch_grid_size
        spatial_attn_upscaled = np.kron(spatial_attn, np.ones((scale, scale)))

        return {
            "spatial_attention": spatial_attn_upscaled,  # (256, 256)
            "raw_attention": last_attn,  # (seq_len, seq_len)
            "patch_attention": spatial_attn,  # (patch_grid, patch_grid)
            "n_image_tokens": n_image_tokens,
            "patch_grid_size": patch_grid_size,
        }

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
