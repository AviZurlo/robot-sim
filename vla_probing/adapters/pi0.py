"""Pi0 adapter for the VLA probing suite.

π0 (3B params) uses PaliGemma as its VLM backbone with a SigLIP vision
encoder, plus a Gemma-300M action expert. Actions are generated via
10-step flow matching denoising. Checkpoint: lerobot/pi0_libero.

EMBODIMENT NOTE:
The pi0_libero checkpoint was fine-tuned on LIBERO (Franka Panda) with
2 camera views (image + image2) and 8D state / 7D action. Our WidowX
MuJoCo scene provides the same image keys and an 8D BridgeData state
vector, making it structurally compatible. However, the visual domain
differs (MuJoCo WidowX vs LIBERO Franka), so results reflect
cross-embodiment transfer properties.

The pi0_base checkpoint expects 3 camera views and 32D state/action —
it's meant as a fine-tuning base and won't produce meaningful zero-shot
actions on any specific embodiment. We default to pi0_libero.
"""

from typing import Any

import numpy as np
import torch

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device


def _install_siglip_check_shim() -> None:
    """Install a compatibility shim for the SigLIP check module.

    lerobot 0.4.3's PI0Pytorch.__init__ imports
    `transformers.models.siglip.check` which only exists in a custom
    transformers fork (branch `fix/lerobot_openpi`). Installing that fork
    breaks other VLA adapters (X-VLA, SmolVLA) that need stock transformers.

    This shim creates a minimal `check` module so PI0Pytorch can instantiate
    with stock transformers >=4.57.
    """
    import importlib
    import sys
    import types

    module_name = "transformers.models.siglip.check"
    if module_name in sys.modules:
        return

    check_module = types.ModuleType(module_name)
    check_module.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules[module_name] = check_module

    # Also register as submodule of the siglip package
    siglip_pkg = importlib.import_module("transformers.models.siglip")
    siglip_pkg.check = check_module


class Pi0Adapter(VLAAdapter):
    """Adapter for π0 (lerobot/pi0_libero checkpoint).

    π0 uses PaliGemma (3B) as VLM backbone with SigLIP vision encoder,
    plus a Gemma-300M action expert connected via interleaved transformer
    layers. Actions are generated via 10-step flow matching denoising.

    The LIBERO checkpoint expects:
        - 2 camera views: observation.images.image, observation.images.image2
          (+ 1 empty camera padded internally)
        - 8D proprioceptive state
        - 7D action output (EE pos + rotation + gripper)
        - chunk_size=50, n_action_steps=10
    """

    model_name = "pi0"

    def __init__(self, checkpoint: str = "lerobot/pi0_libero") -> None:
        self.checkpoint = checkpoint
        self.policy = None
        self.device = None
        self._tokenizer = None
        self._action_dim: int | None = None

    @property
    def action_dim(self) -> int:
        if self._action_dim is not None:
            return self._action_dim
        if self.policy is not None:
            feat = self.policy.config.action_feature
            if feat is not None:
                return feat.shape[0]
        return 7  # LIBERO: 7D (pos3 + rot3 + gripper)

    @property
    def chunk_size(self) -> int:
        if self.policy is not None:
            return self.policy.config.chunk_size
        return 50

    def load_model(self, device: str = "mps") -> None:
        # Patch SigLIP check module before importing PI0Policy.
        # lerobot 0.4.3 expects a custom transformers fork with a
        # `transformers.models.siglip.check` module. We shim it so Pi0
        # works with stock transformers >=4.57.
        _install_siglip_check_shim()

        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        self.device = _get_device(device)
        print(f"Loading π0 ({self.checkpoint}) on {self.device}...")

        policy = PI0Policy.from_pretrained(self.checkpoint)
        policy.to(self.device)
        policy.eval()
        self.policy = policy

        # Cache actual device after placement
        self.device = next(self.policy.parameters()).device

        # Force eager attention on MPS (no flash_attn support)
        self._force_eager_attention()

        # Patch denoise_step to clone KV cache before each step.
        # Stock transformers' GemmaAttention.forward() mutates the
        # DynamicCache even when use_cache=False, causing the cache to
        # grow across denoising iterations. The custom transformers fork
        # (fix/lerobot_openpi) patches this, but we work around it here.
        self._patch_denoise_step()

        # Cache action dim from config
        feat = self.policy.config.action_feature
        if feat is not None:
            self._action_dim = feat.shape[0]

        # Load tokenizer compatible with PaliGemma's Gemma vocabulary.
        # PaliGemma (google/paligemma-3b-pt-224) is gated, so we fall back
        # to unsloth/gemma-2b which shares the same 256K sentencepiece vocab.
        # PaliGemma has 257152 tokens (256000 text + 1152 image), but text
        # tokenization is identical.
        from transformers import AutoTokenizer

        for tokenizer_name in [
            "google/paligemma-3b-pt-224",
            "unsloth/gemma-2b",
        ]:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self._tokenizer.padding_side = "right"
                print(f"  tokenizer: {tokenizer_name} (vocab={self._tokenizer.vocab_size})")
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                "Could not load a Gemma-compatible tokenizer. "
                "Either log in to HF (`huggingface-cli login`) for PaliGemma access, "
                "or ensure `unsloth/gemma-2b` is reachable."
            )

        n_params = sum(p.numel() for p in self.policy.parameters()) / 1e6
        print(f"π0 loaded: {n_params:.1f}M params on {self.device}")
        print(f"  action_dim={self.action_dim}, chunk_size={self.chunk_size}")

    def _patch_denoise_step(self) -> None:
        """Patch PI0Pytorch.denoise_step to clone KV cache before use.

        Stock transformers' DynamicCache is mutated by GemmaAttention.forward
        even when use_cache=False. This causes the cache to grow across
        denoising steps (816 → 867 → 918 → ...), leading to dimension
        mismatches. We wrap denoise_step to deep-copy the cache each time.
        """
        import copy

        original_denoise = self.policy.model.denoise_step

        def patched_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep):
            kv_copy = copy.deepcopy(past_key_values)
            return original_denoise(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=kv_copy,
                x_t=x_t,
                timestep=timestep,
            )

        self.policy.model.denoise_step = patched_denoise

    def _force_eager_attention(self) -> None:
        """Force eager attention implementation (no flash_attn on MPS)."""
        model = self.policy.model
        # PaliGemma language model
        pali_lm = model.paligemma_with_expert.paligemma.language_model
        if hasattr(pali_lm, "config"):
            pali_lm.config._attn_implementation = "eager"
        # Gemma action expert
        expert = model.paligemma_with_expert.gemma_expert
        if hasattr(expert, "model") and hasattr(expert.model, "config"):
            expert.model.config._attn_implementation = "eager"

    def _prepare_batch(self, inp: VLAInput) -> dict[str, torch.Tensor]:
        """Convert VLAInput to the batch dict PI0Policy expects.

        PI0Policy expects:
        - observation.images.image: (B, 3, H, W) float [0, 1]
        - observation.images.image2: (B, 3, H, W) float [0, 1]
        - observation.state: (B, state_dim) float
        - observation.language.tokens: (B, max_len) long
        - observation.language.attention_mask: (B, max_len) long
        """
        # Images: (H, W, 3) uint8 -> (1, 3, H, W) float [0, 1]
        imgs = []
        for img in inp.images[:2]:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(t.unsqueeze(0).to(self.device))

        # Pad to 2 images if only 1 provided
        while len(imgs) < 2:
            imgs.append(torch.zeros_like(imgs[0]))

        # State: (state_dim,) -> (1, state_dim)
        # PI0's prepare_state() handles padding to max_state_dim=32
        state = torch.from_numpy(inp.proprio).float().unsqueeze(0).to(self.device)

        # Language: tokenize with PaliGemma tokenizer
        # Pi0 expects task text ending with newline
        prompt = inp.prompt if inp.prompt.endswith("\n") else f"{inp.prompt}\n"
        tokens = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
        )

        return {
            "observation.images.image": imgs[0],
            "observation.images.image2": imgs[1],
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].bool().to(self.device),
        }

    def predict_action(self, inp: VLAInput) -> VLAOutput:
        batch = self._prepare_batch(inp)
        with torch.no_grad():
            self.policy.reset()
            actions = self.policy.predict_action_chunk(batch)

        # (1, chunk_size, action_dim) -> (chunk_size, action_dim)
        actions_np = actions[0].cpu().float().numpy()
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
        """Extract attention maps from π0's PaliGemma VLM backbone.

        Hooks into GemmaAttention.forward() to capture self-attention
        weights during the prefix forward pass (image + language tokens).

        GemmaDecoderLayer discards attention weights, so we register
        forward hooks on each layer's self_attn module to intercept them.
        """
        batch = self._prepare_batch(inp)
        attn_weights_captured: list[torch.Tensor] = []

        pali_lm = self.policy.model.paligemma_with_expert.paligemma.language_model

        # Force eager attention — required for attention weight output
        pali_lm.config._attn_implementation = "eager"

        # Hook into each GemmaAttention module to capture weights.
        # GemmaAttention.forward() returns (attn_output, attn_weights),
        # but GemmaDecoderLayer discards attn_weights with `_`.
        hooks = []
        for layer in pali_lm.layers:
            def make_hook():
                def hook_fn(module, args, kwargs, output):
                    # output = (attn_output, attn_weights)
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_weights_captured.append(output[1].detach().cpu())
                return hook_fn
            h = layer.self_attn.register_forward_hook(make_hook(), with_kwargs=True)
            hooks.append(h)

        try:
            with torch.no_grad():
                self.policy.reset()

                images, img_masks = self.policy._preprocess_images(batch)
                lang_tokens = batch["observation.language.tokens"]
                lang_masks = batch["observation.language.attention_mask"]

                from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.policy.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks
                    )
                )

                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                att_2d_masks_4d = self.policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)

                # Forward through GemmaModel — handles RoPE and normalizer internally
                pali_lm(
                    inputs_embeds=prefix_embs,
                    attention_mask=att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    output_attentions=True,
                )
        finally:
            for h in hooks:
                h.remove()

        if not attn_weights_captured:
            return self._fallback_attention()

        # Use last layer attention, average over heads
        # Shape: (1, n_heads, seq_len, seq_len) -> (seq_len, seq_len)
        last_attn = attn_weights_captured[-1][0].float().mean(dim=0).numpy()

        # Determine image token count from SigLIP vision encoder
        with torch.no_grad():
            test_img = images[0]
            img_emb = self.policy.model.paligemma_with_expert.embed_image(test_img)
            n_image_tokens_per_view = img_emb.shape[1]

        # Total image tokens = tokens_per_view × number of real camera views
        n_real_cameras = sum(1 for m in img_masks if m.any())
        n_image_tokens = n_image_tokens_per_view * n_real_cameras

        patch_grid_size = int(np.sqrt(n_image_tokens_per_view))
        n_spatial = patch_grid_size * patch_grid_size

        # Average attention TO first camera's image tokens across all query positions
        seq_len = last_attn.shape[0]
        if n_spatial > 0 and n_spatial <= seq_len:
            img_attn = last_attn[:, :n_spatial].mean(axis=0)
            spatial_attn = img_attn.reshape(patch_grid_size, patch_grid_size)

            # Upscale to image resolution (256x256)
            scale = max(1, 256 // patch_grid_size)
            spatial_attn_upscaled = np.kron(spatial_attn, np.ones((scale, scale)))
            # Crop/pad to exactly 256x256
            h, w = spatial_attn_upscaled.shape
            result = np.zeros((256, 256))
            result[:min(h, 256), :min(w, 256)] = spatial_attn_upscaled[:256, :256]
        else:
            return self._fallback_attention()

        return {
            "spatial_attention": result,
            "raw_attention": last_attn,
            "patch_attention": spatial_attn,
            "n_image_tokens": n_image_tokens,
            "patch_grid_size": patch_grid_size,
        }

    def _fallback_attention(self) -> dict[str, Any]:
        """Return uniform attention when extraction fails."""
        return {
            "spatial_attention": np.ones((256, 256)) / (256 * 256),
            "raw_attention": np.zeros((1, 1)),
            "patch_attention": np.ones((1, 1)),
            "n_image_tokens": 0,
            "patch_grid_size": 1,
        }

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
