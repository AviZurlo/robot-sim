"""OpenVLA-7B adapter for the VLA probing suite.

OpenVLA uses a Prismatic VLM backbone (DINOv2 + SigLIP vision encoders)
feeding into Llama-2 7B. Unlike flow-matching VLAs, actions are generated
as discretized text tokens from the language model's vocabulary.

Architecture:
    Image (224x224) -> DINOv2 ViT-L/14 + SigLIP ViT-So400M/14
        -> fused-gelu-mlp projector -> patch embeddings
    Prompt -> Llama-2 tokenizer -> text embeddings
    [BOS] [vision_patches...] [text_tokens...] -> Llama-2 7B -> 7 action tokens

Action format: 7-DoF end-effector deltas [x, y, z, roll, pitch, yaw, gripper]
Output is 1 timestep (autoregressive, not chunked like flow-matching models).
"""

from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device


# Standard OpenVLA prompt template
OPENVLA_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


def _check_mps_memory(required_gb: float = 14.0) -> bool:
    """Check if MPS has enough memory for the model.

    On Apple Silicon, unified memory is shared between CPU and GPU.
    OpenVLA-7B at fp16 needs ~14GB for weights alone, plus activations.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        # Apple Silicon total memory check via platform
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024**3)
        # Need headroom for OS + activations
        return total_gb >= required_gb + 6
    except Exception:
        return False


class OpenVLAAdapter(VLAAdapter):
    """Adapter for OpenVLA-7B (openvla/openvla-7b checkpoint).

    OpenVLA is the only model in the probing suite that outputs actions
    as text tokens, enabling VLM scene querying as a unique probe.

    Key differences from X-VLA:
        - Single image input (not dual-camera)
        - No proprioception input used
        - 7-DoF action output (not 20D packed timesteps)
        - 1 action step per inference (not 30-step chunks)
        - Deterministic output (no flow matching stochasticity)
        - Can generate free-form text (VLM querying)
    """

    model_name = "openvla"

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None

    @property
    def action_dim(self) -> int:
        return 7  # [x, y, z, roll, pitch, yaw, gripper]

    @property
    def chunk_size(self) -> int:
        return 1  # OpenVLA predicts one timestep at a time

    def load_model(self, device: str = "mps") -> None:
        """Load OpenVLA-7B in fp16.

        Attempts MPS first (24GB unified memory should fit 7B at fp16),
        falls back to CPU if MPS can't handle it.
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Determine device — MPS needs ~18GB peak, so 24GB should work
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

        print(f"Loading OpenVLA-7B on {self.device} (fp16)...")

        # Load processor (handles image preprocessing + tokenization)
        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
        )

        # Load model in fp16 — bfloat16 not fully supported on MPS
        model_dtype = torch.float16
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model.to(self.device)
            self.model.eval()
        except (RuntimeError, torch.mps.OutOfMemoryError) if hasattr(torch, "mps") else RuntimeError as e:
            if self.device.type == "mps":
                print(f"MPS OOM ({e}), falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    "openvla/openvla-7b",
                    torch_dtype=torch.float32,  # CPU works better with fp32
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.model.to(self.device)
                self.model.eval()
            else:
                raise

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(
            f"OpenVLA loaded: {n_params:.1f}B params on {self.device} "
            f"({next(self.model.parameters()).dtype})"
        )

    def _prepare_inputs(
        self, inp: VLAInput, prompt_override: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Convert VLAInput to OpenVLA processor inputs.

        OpenVLA takes a single image and a text prompt.
        The proprioceptive state is not used (OpenVLA doesn't take proprio).
        """
        # OpenVLA uses a single camera view
        image = PILImage.fromarray(inp.images[0])

        # Format the prompt using OpenVLA's template
        instruction = prompt_override or inp.prompt
        prompt = OPENVLA_PROMPT.format(instruction=instruction)

        # Process through the Prismatic processor
        inputs = self.processor(prompt, image)
        model_dtype = next(self.model.parameters()).dtype
        return inputs.to(self.device, dtype=model_dtype)

    def predict_action(self, inp: VLAInput) -> VLAOutput:
        """Run inference and return 7-DoF action prediction.

        Uses the model's built-in predict_action method which handles
        token generation, discretized bin decoding, and unnormalization.
        """
        inputs = self._prepare_inputs(inp)

        with torch.no_grad():
            # predict_action returns a numpy array of shape (7,)
            action = self.model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
            )

        # Reshape to (1, 7) to match VLAOutput convention (chunk_size, action_dim)
        return VLAOutput(
            actions=action.reshape(1, -1),
            raw_output={"unnorm_key": "bridge_orig"},
        )

    def query_vlm(
        self,
        images: list[np.ndarray],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Query OpenVLA as a VLM — ask it to describe the scene.

        This is unique to OpenVLA: since actions are text tokens from the
        same output head as language, we can prompt it for free-form text.
        The model may not produce coherent scene descriptions (it was
        fine-tuned for actions), but testing this is Probe 0.

        Args:
            images: List of RGB images (uses first image).
            prompt: Free-form text prompt (e.g. "Describe what you see").
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        image = PILImage.fromarray(images[0])

        # Use a conversational prompt format instead of the action template
        text_prompt = f"In: {prompt}\nOut:"
        inputs = self.processor(text_prompt, image)
        model_dtype = next(self.model.parameters()).dtype
        inputs = inputs.to(self.device, dtype=model_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens (skip the input)
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0, input_len:]
        response = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()

    def get_attention(self, inp: VLAInput) -> dict[str, np.ndarray]:
        """Extract attention maps from Llama-2 layers.

        OpenVLA's Llama-2 backbone uses causal self-attention over the
        sequence [BOS, vision_patches..., text_tokens...]. We extract
        the last layer's attention and isolate the vision-patch region
        to produce spatial attention maps.

        The fused DINOv2+SigLIP backbone at 224px with patch size 14
        produces ~256 patches per encoder. After fusion and projection,
        the exact count depends on the projector architecture.
        """
        inputs = self._prepare_inputs(inp)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # Get the last layer's attention: (1, n_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1][0]  # (n_heads, seq_len, seq_len)

        # Average over attention heads
        avg_attn = last_attn.mean(dim=0).cpu().float().numpy()  # (seq_len, seq_len)

        # Determine vision token positions
        # Layout: [BOS] [vision_patch_0...vision_patch_N] [text_token_1...text_token_M]
        input_ids = inputs["input_ids"]
        seq_len = avg_attn.shape[0]
        n_text_tokens = input_ids.shape[1]

        # Vision tokens are inserted after BOS (position 0)
        # Total seq = 1 (BOS) + n_vision + n_text_remaining
        n_vision_tokens = seq_len - n_text_tokens
        if n_vision_tokens <= 0:
            # Fallback: no vision tokens detected, return uniform attention
            n_vision_tokens = 256  # reasonable default
            print(
                f"WARNING: Could not determine vision token count "
                f"(seq_len={seq_len}, n_text={n_text_tokens}). Using {n_vision_tokens}."
            )

        # Vision tokens span positions [1, 1+n_vision_tokens)
        vision_start = 1
        vision_end = vision_start + n_vision_tokens

        # Attention FROM text tokens TO vision tokens
        # This shows which image regions the language model attends to
        text_start = vision_end
        if text_start < seq_len:
            # (n_text_tokens, n_vision_tokens) — how text attends to vision
            text_to_vision = avg_attn[text_start:, vision_start:vision_end]
            # Average across text query positions
            vision_attn_1d = text_to_vision.mean(axis=0)
        else:
            # All-to-vision attention as fallback
            vision_attn_1d = avg_attn[:, vision_start:vision_end].mean(axis=0)

        # Reshape to 2D spatial grid
        # The projector may change the patch count, but typically:
        # DINOv2: 16x16=256, SigLIP: 16x16=256, fused: varies
        patch_grid_size = int(np.sqrt(n_vision_tokens))
        if patch_grid_size * patch_grid_size != n_vision_tokens:
            # Non-square token count — try closest square crop
            patch_grid_size = int(np.ceil(np.sqrt(n_vision_tokens)))
            padded = np.zeros(patch_grid_size * patch_grid_size)
            padded[: len(vision_attn_1d)] = vision_attn_1d
            vision_attn_1d = padded

        spatial_attn = vision_attn_1d.reshape(patch_grid_size, patch_grid_size)

        # Upscale to 256x256 (match scene render resolution)
        target_size = 256
        scale = max(1, target_size // patch_grid_size)
        spatial_attn_upscaled = np.kron(spatial_attn, np.ones((scale, scale)))
        # Trim to exactly target_size if needed
        spatial_attn_upscaled = spatial_attn_upscaled[:target_size, :target_size]

        return {
            "spatial_attention": spatial_attn_upscaled,  # (256, 256)
            "raw_attention": avg_attn,  # (seq_len, seq_len)
            "patch_attention": spatial_attn,  # (grid, grid)
            "n_image_tokens": n_vision_tokens,
            "patch_grid_size": patch_grid_size,
        }

    def reset(self) -> None:
        """No internal state to reset (OpenVLA is stateless)."""
