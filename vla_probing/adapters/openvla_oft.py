"""OpenVLA-OFT adapter for the VLA probing suite.

OpenVLA-OFT (Optimized Fine-Tuning) improves on OpenVLA by replacing the
discrete token action output with a continuous MLP action head (L1 regression).
It also adds dual-camera support and proprioception input.

Architecture:
    Image_primary (224x224) + Image_wrist (224x224) -> DINOv2 + SigLIP
        -> fused-gelu-mlp projector -> patch embeddings
    Proprioception (8D) -> ProprioProjector -> embedding
    Prompt -> Llama-2 tokenizer -> text embeddings
    [BOS] [vision_patches...] [proprio] [text_tokens...] [action_tokens...]
        -> Llama-2 7B -> action hidden states -> MLP action head
        -> 8-step action chunks of 7D

Action format: 7-DoF end-effector deltas [x, y, z, roll, pitch, yaw, gripper]
Output is 8 timesteps (action chunks) per inference call.

Checkpoint: moojink/openvla-7b-oft-finetuned-libero-spatial
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device

# Vendor directory contains a minimal prismatic shim package with just
# the modules needed by the HF Hub modeling code (constants, train_utils,
# action_heads, projectors). This avoids installing the full openvla-oft
# repo with its heavy dependencies (tensorflow, custom transformers fork, etc.)
_VENDOR_DIR = str(Path(__file__).resolve().parent.parent.parent / "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# Constants from LIBERO config
_NUM_ACTIONS_CHUNK = 8
_ACTION_DIM = 7
_PROPRIO_DIM = 8

# Checkpoint and normalization
_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
_UNNORM_KEY = "libero_spatial_no_noops"

# Standard OpenVLA prompt template
_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def _check_mps_memory(required_gb: float = 14.0) -> bool:
    """Check if MPS has enough memory for the model."""
    if not torch.backends.mps.is_available():
        return False
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024**3)
        return total_gb >= required_gb + 6
    except Exception:
        return False


def _load_action_head(llm_dim: int, device: torch.device, dtype: torch.dtype):
    """Load the L1 regression action head from HF Hub."""
    from prismatic.models.action_heads import L1RegressionActionHead

    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=_ACTION_DIM
    )

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_CHECKPOINT,
        filename="action_head--150000_checkpoint.pt",
    )
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    # Strip DDP "module." prefix if present
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.removeprefix("module.")] = v
    action_head.load_state_dict(clean_sd)
    action_head = action_head.to(dtype).to(device)
    action_head.eval()
    return action_head


def _load_proprio_projector(llm_dim: int, device: torch.device, dtype: torch.dtype):
    """Load the proprioception projector from HF Hub."""
    from prismatic.models.projectors import ProprioProjector

    projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=_PROPRIO_DIM)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_CHECKPOINT,
        filename="proprio_projector--150000_checkpoint.pt",
    )
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.removeprefix("module.")] = v
    projector.load_state_dict(clean_sd)
    projector = projector.to(dtype).to(device)
    projector.eval()
    return projector


def _normalize_proprio(proprio: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Normalize proprioception to [-1, 1] using q01/q99 bounds (LIBERO style)."""
    proprio_high = np.array(norm_stats["q99"])
    proprio_low = np.array(norm_stats["q01"])
    normalized = 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1
    return np.clip(normalized, -1.0, 1.0)


def _resize_image(img: np.ndarray, size: int = 224) -> PILImage.Image:
    """Resize image to the expected input size and return as PIL."""
    pil_img = PILImage.fromarray(img).convert("RGB")
    if pil_img.size != (size, size):
        pil_img = pil_img.resize((size, size), PILImage.LANCZOS)
    return pil_img


class OpenVLAOFTAdapter(VLAAdapter):
    """Adapter for OpenVLA-OFT (LIBERO spatial checkpoint).

    Key differences from vanilla OpenVLA:
        - Continuous actions via MLP action head (not discrete tokens)
        - Dual camera inputs (primary + wrist)
        - Proprioception input (8D)
        - 8-step action chunks (not single-step)
        - No temperature needed (deterministic L1 regression head)

    For variance measurement, the model still produces some variance
    from random seed changes affecting internal torch operations.
    """

    model_name = "openvla_oft"

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self.device = None
        self._norm_stats = None

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM  # 7: [x, y, z, roll, pitch, yaw, gripper]

    @property
    def chunk_size(self) -> int:
        return _NUM_ACTIONS_CHUNK  # 8 action steps per inference

    def load_model(self, device: str = "mps") -> None:
        """Load OpenVLA-OFT: base VLA + action head + proprio projector."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Determine device
        if device == "mps" and _check_mps_memory(required_gb=14.0):
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if device == "mps":
                print(
                    "WARNING: MPS requested but may not have enough memory. "
                    "Attempting MPS anyway, will fall back to CPU on OOM."
                )
                self.device = _get_device("mps")
            else:
                self.device = torch.device("cpu")

        # MPS has issues with fp16 matmul in the action head (NDArray dtype mismatch).
        # Use fp32 on MPS (24GB unified memory fits 7B at fp32 ~14GB + activations).
        # Use fp16 on CUDA where it works reliably.
        if self.device.type == "mps":
            model_dtype = torch.float32
        else:
            model_dtype = torch.float16
        dtype_label = "fp32" if model_dtype == torch.float32 else "fp16"
        print(f"Loading OpenVLA-OFT on {self.device} ({dtype_label})...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            _CHECKPOINT, trust_remote_code=True
        )

        # Load base VLA model
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                _CHECKPOINT,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            # Set number of images in input (primary + wrist = 2)
            self.model.vision_backbone.set_num_images_in_input(2)
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM ({e}), falling back to CPU fp32...")
                self.device = torch.device("cpu")
                model_dtype = torch.float32
                self.model = AutoModelForVision2Seq.from_pretrained(
                    _CHECKPOINT,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                self.model.vision_backbone.set_num_images_in_input(2)
                self.model.to(self.device)
                self.model.eval()
            else:
                raise

        # Load dataset statistics for normalization
        from huggingface_hub import hf_hub_download

        stats_path = hf_hub_download(
            repo_id=_CHECKPOINT, filename="dataset_statistics.json"
        )
        with open(stats_path) as f:
            self._norm_stats = json.load(f)
        self.model.norm_stats = self._norm_stats

        # Load action head and proprio projector
        llm_dim = self.model.llm_dim
        self.action_head = _load_action_head(llm_dim, self.device, model_dtype)
        self.proprio_projector = _load_proprio_projector(
            llm_dim, self.device, model_dtype
        )

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(
            f"OpenVLA-OFT loaded: {n_params:.1f}B params + action head + proprio projector "
            f"on {self.device} ({model_dtype})"
        )

    def _prepare_inputs(self, inp: VLAInput) -> tuple[dict, np.ndarray]:
        """Convert VLAInput to model inputs.

        Returns (processed_inputs, normalized_proprio).
        """
        # Process primary and wrist images
        primary_img = _resize_image(inp.images[0])
        wrist_img = _resize_image(inp.images[1] if len(inp.images) > 1 else inp.images[0])

        # Format prompt
        prompt = _PROMPT_TEMPLATE.format(instruction=inp.prompt)

        # Process primary image through processor
        model_dtype = next(self.model.parameters()).dtype
        inputs = self.processor(prompt, primary_img).to(self.device, dtype=model_dtype)

        # Process wrist image and concatenate pixel values
        wrist_inputs = self.processor(prompt, wrist_img).to(self.device, dtype=model_dtype)
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )

        # Normalize proprioception
        proprio_stats = self._norm_stats[_UNNORM_KEY]["proprio"]
        normalized_proprio = _normalize_proprio(inp.proprio, proprio_stats)

        return inputs, normalized_proprio

    def predict_action(self, inp: VLAInput) -> VLAOutput:
        """Run inference and return 8-step action chunk prediction."""
        inputs, normalized_proprio = self._prepare_inputs(inp)

        with torch.inference_mode():
            actions, _ = self.model.predict_action(
                **inputs,
                unnorm_key=_UNNORM_KEY,
                do_sample=False,
                proprio=normalized_proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
            )

        # actions shape: (NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return VLAOutput(
            actions=actions.reshape(-1, _ACTION_DIM),
            raw_output={"unnorm_key": _UNNORM_KEY, "chunk_size": _NUM_ACTIONS_CHUNK},
        )

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Extract attention maps from Llama-2 layers.

        We bypass the OFT model's forward() (which requires training-style
        labels/action masks) and instead manually construct the input embeddings
        (vision patches + text tokens) and run the language model directly with
        output_attentions=True.
        """
        inputs, _ = self._prepare_inputs(inp)

        with torch.no_grad():
            # Get vision features (projected patch embeddings)
            pixel_values = inputs["pixel_values"]
            # The model's _process_vision_features builds projected patches
            # from pixel values using the vision backbone + projector
            patch_features = self.model.vision_backbone(pixel_values)
            projected_patches = self.model.projector(patch_features)

            # Get text embeddings
            input_embeds = self.model.get_input_embeddings()(inputs["input_ids"])

            # Merge: [BOS_embed] [projected_patches] [text_embeds_after_BOS]
            bos_embed = input_embeds[:, :1, :]  # BOS token
            text_rest = input_embeds[:, 1:, :]  # rest of text tokens
            merged = torch.cat([bos_embed, projected_patches, text_rest], dim=1)

            # Build attention mask for merged sequence
            seq_len = merged.shape[1]
            attn_mask = torch.ones(1, seq_len, device=self.device, dtype=merged.dtype)

            # Run through the language model
            lm_outputs = self.model.language_model(
                inputs_embeds=merged,
                attention_mask=attn_mask,
                output_attentions=True,
                return_dict=True,
            )

        # Get last layer attention: (1, n_heads, seq_len, seq_len)
        last_attn = lm_outputs.attentions[-1][0]  # (n_heads, seq_len, seq_len)
        avg_attn = last_attn.mean(dim=0).cpu().float().numpy()

        # Determine vision token positions
        # Layout: [BOS] [vision_patches...] [text_tokens_after_BOS...]
        n_vision_tokens = projected_patches.shape[1]
        seq_len = avg_attn.shape[0]

        vision_start = 1
        vision_end = vision_start + n_vision_tokens

        # Attention from text tokens to vision tokens
        text_start = vision_end
        if text_start < seq_len:
            text_to_vision = avg_attn[text_start:, vision_start:vision_end]
            vision_attn_1d = text_to_vision.mean(axis=0)
        else:
            vision_attn_1d = avg_attn[:, vision_start:vision_end].mean(axis=0)

        # Use only primary image patches (first half) for spatial attention
        n_patches_per_image = n_vision_tokens // 2
        primary_attn = vision_attn_1d[:n_patches_per_image]

        patch_grid_size = int(np.sqrt(n_patches_per_image))
        if patch_grid_size * patch_grid_size != n_patches_per_image:
            patch_grid_size = int(np.ceil(np.sqrt(n_patches_per_image)))
            padded = np.zeros(patch_grid_size * patch_grid_size)
            padded[: len(primary_attn)] = primary_attn
            primary_attn = padded

        spatial_attn = primary_attn.reshape(patch_grid_size, patch_grid_size)

        target_size = 256
        scale = max(1, target_size // patch_grid_size)
        spatial_attn_upscaled = np.kron(spatial_attn, np.ones((scale, scale)))
        spatial_attn_upscaled = spatial_attn_upscaled[:target_size, :target_size]

        return {
            "spatial_attention": spatial_attn_upscaled,
            "raw_attention": avg_attn,
            "patch_attention": spatial_attn,
            "n_image_tokens": n_vision_tokens,
            "patch_grid_size": patch_grid_size,
        }

    def reset(self) -> None:
        """No internal state to reset (stateless inference)."""
