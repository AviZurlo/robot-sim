"""SmolVLA adapter for the VLA probing suite.

SmolVLA (~0.5B params) uses SmolVLM2-500M as its VLM backbone with a
smaller action expert connected via cross-attention, and a flow matching
action head. Checkpoint: lerobot/smolvla_base.

CROSS-EMBODIMENT NOTE:
SmolVLA was trained on community robot data (SO-100, LeKiwi) — NOT on
WidowX / BridgeData. The WidowX MuJoCo scene used by this probing suite
produces out-of-distribution inputs. Results should be interpreted as
*cross-embodiment zero-shot transfer* probes, testing architectural
properties (attention patterns, null-action compliance, stochasticity)
rather than task-specific performance.
"""

from typing import Any

import numpy as np
import torch

from vla_probing.adapter import VLAAdapter, VLAInput, VLAOutput, _get_device


class SmolVLAAdapter(VLAAdapter):
    """Adapter for SmolVLA (lerobot/smolvla_base checkpoint).

    SmolVLA uses SmolVLM2-500M-Video-Instruct as VLM backbone with a
    SigLIP vision encoder, plus a smaller action expert (75% width)
    connected via cross-attention. Actions are generated via 10-step
    flow matching denoising.

    The pretrained checkpoint was trained on SO-100 community data.
    Action format depends on the training embodiment — the base
    checkpoint outputs 32D padded vectors, trimmed to the actual
    action dimension stored in the config.
    """

    model_name = "smolvla"

    def __init__(self) -> None:
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
        return 6  # fallback: SO-100 typically uses 6D actions

    @property
    def chunk_size(self) -> int:
        if self.policy is not None:
            return self.policy.config.chunk_size
        return 50

    def load_model(self, device: str = "mps") -> None:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = _get_device(device)

        print(f"Loading SmolVLA on {self.device}...")

        # SmolVLA's from_pretrained reads config.device to place weights.
        # We patch via config after loading.
        policy = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base",
        )
        policy.to(self.device)
        policy.eval()
        self.policy = policy

        # Cache actual device after placement
        self.device = next(self.policy.parameters()).device

        # Cache action dim from config
        feat = self.policy.config.action_feature
        if feat is not None:
            self._action_dim = feat.shape[0]

        # Cache tokenizer from the VLM backbone processor
        self._tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer

        n_params = sum(p.numel() for p in self.policy.parameters()) / 1e6
        print(f"SmolVLA loaded: {n_params:.1f}M params on {self.device}")
        print(f"  action_dim={self.action_dim}, chunk_size={self.chunk_size}")

    def _prepare_batch(self, inp: VLAInput) -> dict[str, torch.Tensor]:
        """Convert VLAInput to the batch dict SmolVLAPolicy expects.

        SmolVLA's config defines image feature keys (e.g. camera1, camera2, camera3).
        We map our WidowX scene images to the first camera and pad the rest.
        """
        # Discover image feature keys from the pretrained config
        img_keys = list(self.policy.config.image_features.keys())

        batch: dict[str, torch.Tensor] = {}

        # Map input images to config camera keys; pad any extra cameras
        for i, key in enumerate(img_keys):
            if i < len(inp.images):
                img = inp.images[i]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            else:
                # Pad missing cameras with the first image (avoids NaN from zeros)
                img = inp.images[0]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[key] = img_t.unsqueeze(0).to(self.device)

        # State: (state_dim,) -> (1, state_dim)
        # SmolVLA's prepare_state() handles padding to max_state_dim=32
        state = torch.from_numpy(inp.proprio).float().unsqueeze(0).to(self.device)
        batch["observation.state"] = state

        # Language: tokenize with SmolVLM2 tokenizer
        # SmolVLA expects task text ending with newline
        prompt = inp.prompt if inp.prompt.endswith("\n") else f"{inp.prompt}\n"
        tokens = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
        )
        batch["observation.language.tokens"] = tokens["input_ids"].to(self.device)
        batch["observation.language.attention_mask"] = (
            tokens["attention_mask"].bool().to(self.device)
        )

        return batch

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
        """Extract attention maps from SmolVLA's VLM backbone.

        Monkey-patches eager_attention_forward to capture the softmax
        attention probabilities (already computed with GQA expansion).
        """
        batch = self._prepare_batch(inp)

        vlm_expert = self.policy.model.vlm_with_expert
        original_eager = vlm_expert.eager_attention_forward
        attn_weights_captured: list[torch.Tensor] = []

        def capturing_eager(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            """Drop-in replacement that captures probs before returning."""
            from torch import nn as _nn

            num_att_heads = vlm_expert.num_attention_heads
            num_kv_heads = vlm_expert.num_key_value_heads
            num_kv_groups = num_att_heads // num_kv_heads
            seq_len = key_states.shape[1]

            # GQA expansion (same as original)
            k = key_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim
            ).reshape(batch_size, seq_len, num_att_heads, head_dim)
            v = value_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim
            ).reshape(batch_size, seq_len, num_att_heads, head_dim)

            q = query_states.to(torch.float32).transpose(1, 2)
            k = k.to(torch.float32).transpose(1, 2)

            att_weights = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
            big_neg = torch.finfo(att_weights.dtype).min
            att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = _nn.functional.softmax(att_weights, dim=-1)

            # Capture
            attn_weights_captured.append(probs.detach().cpu())

            probs = probs.to(dtype=v.dtype)
            out = torch.matmul(probs, v.permute(0, 2, 1, 3))
            out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, num_att_heads * head_dim)
            return out

        vlm_expert.eager_attention_forward = capturing_eager

        try:
            with torch.no_grad():
                self.policy.reset()
                images, img_masks = self.policy.prepare_images(batch)
                state = self.policy.prepare_state(batch)
                lang_tokens = batch["observation.language.tokens"]
                lang_masks = batch["observation.language.attention_mask"]

                from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.policy.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks, state=state
                    )
                )
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

                # fill_kv_cache=True routes all layers through forward_attn_layer
                # (self-attention only), avoiding cross-attn layers that need
                # the expert input we don't have during prefix-only pass.
                vlm_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                    fill_kv_cache=True,
                )
        finally:
            vlm_expert.eager_attention_forward = original_eager

        if not attn_weights_captured:
            # Fallback: return uniform attention
            return {
                "spatial_attention": np.ones((256, 256)) / (256 * 256),
                "raw_attention": np.zeros((1, 1)),
                "patch_attention": np.ones((1, 1)),
                "n_image_tokens": 0,
                "patch_grid_size": 1,
            }

        # Use last layer attention, average over heads
        # Shape: (1, n_heads, seq_len, seq_len) -> (seq_len, seq_len)
        last_attn = attn_weights_captured[-1][0].mean(dim=0).numpy()

        # Determine image token count from embed_prefix structure.
        # SigLIP produces a fixed number of patch tokens per image.
        # After connector resampling in SmolVLM2, the count depends on
        # the model's vision config. We estimate from the embedding shapes.
        with torch.no_grad():
            test_img = images[0]
            img_emb = self.policy.model.vlm_with_expert.embed_image(test_img)
            n_image_tokens = img_emb.shape[1]

        patch_grid_size = int(np.sqrt(n_image_tokens))
        if patch_grid_size * patch_grid_size != n_image_tokens:
            # Non-square patch grid — use closest approximation
            patch_grid_size = max(1, patch_grid_size)

        n_spatial = min(patch_grid_size * patch_grid_size, n_image_tokens)

        # Average attention TO image tokens across all query positions
        seq_len = last_attn.shape[0]
        if n_spatial > 0 and n_spatial <= seq_len:
            img_attn = last_attn[:, :n_spatial].mean(axis=0)

            if n_spatial == patch_grid_size * patch_grid_size:
                spatial_attn = img_attn.reshape(patch_grid_size, patch_grid_size)
            else:
                # Reshape to closest square
                spatial_attn = img_attn[:patch_grid_size * patch_grid_size].reshape(
                    patch_grid_size, patch_grid_size
                )

            # Upscale to image resolution (256x256)
            scale = max(1, 256 // patch_grid_size)
            spatial_attn_upscaled = np.kron(spatial_attn, np.ones((scale, scale)))
            # Crop/pad to exactly 256x256
            h, w = spatial_attn_upscaled.shape
            result = np.zeros((256, 256))
            result[:min(h, 256), :min(w, 256)] = spatial_attn_upscaled[:256, :256]
        else:
            result = np.ones((256, 256)) / (256 * 256)
            spatial_attn = np.ones((1, 1))

        return {
            "spatial_attention": result,
            "raw_attention": last_attn,
            "patch_attention": spatial_attn,
            "n_image_tokens": n_image_tokens,
            "patch_grid_size": patch_grid_size,
        }

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
