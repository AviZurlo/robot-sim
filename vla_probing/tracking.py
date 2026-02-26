"""W&B experiment tracking for VLA probing suite.

Handles logging probe results, metrics, and artifacts to
Weights & Biases with the schema defined in the project doc.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProbeResult:
    """Standardized result from a single probe run."""

    model: str  # "xvla" | "pi0" | "smolvla" | "openvla"
    embodiment: str  # "widowx" | "libero_franka" | "so100"
    probe: str  # "baseline" | "spatial_symmetry" | etc.
    probe_variant: str  # specific variant description
    seed: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """W&B experiment tracker for VLA probing."""

    def __init__(
        self,
        project: str = "vla-comparison",
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.enabled = enabled
        self._run = None
        self._offline_results: list[dict] = []

    def init_run(
        self,
        name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a W&B run."""
        if not self.enabled:
            return
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                name=name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
        except Exception as e:
            print(f"W&B init failed ({e}), logging offline")
            self.enabled = False

    def log_probe_result(self, result: ProbeResult) -> None:
        """Log a probe result to W&B (or offline storage)."""
        log_data = {
            "model": result.model,
            "embodiment": result.embodiment,
            "probe": result.probe,
            "probe_variant": result.probe_variant,
            "seed": result.seed,
            **{f"metrics/{k}": v for k, v in result.metrics.items()},
        }

        if self.enabled and self._run is not None:
            import wandb

            # Log scalar metrics
            self._run.log(log_data)

            # Log image artifacts
            for name, artifact in result.artifacts.items():
                if isinstance(artifact, np.ndarray) and artifact.ndim == 3:
                    self._run.log(
                        {f"artifacts/{name}": wandb.Image(artifact)}
                    )
                elif isinstance(artifact, (str, Path)):
                    path = Path(artifact)
                    if path.exists() and path.suffix in (".png", ".jpg", ".html"):
                        self._run.log(
                            {f"artifacts/{name}": wandb.Image(str(path))}
                            if path.suffix in (".png", ".jpg")
                            else {f"artifacts/{name}": wandb.Html(str(path))}
                        )
        else:
            self._offline_results.append(
                {**log_data, "artifacts": list(result.artifacts.keys())}
            )

    def log_trajectory_plot(
        self,
        trajectory_xyz: np.ndarray,
        name: str = "trajectory",
    ) -> None:
        """Log an interactive 3D trajectory plot."""
        if not self.enabled or self._run is None:
            return
        try:
            import plotly.graph_objects as go
            import wandb

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=trajectory_xyz[:, 0],
                        y=trajectory_xyz[:, 1],
                        z=trajectory_xyz[:, 2],
                        mode="markers+lines",
                        marker=dict(
                            size=4,
                            color=np.arange(len(trajectory_xyz)),
                            colorscale="RdYlGn_r",
                        ),
                        line=dict(width=2, color="gray"),
                    )
                ]
            )
            fig.update_layout(
                title=name,
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                ),
                width=600,
                height=500,
            )
            self._run.log({f"trajectories/{name}": wandb.Plotly(fig)})
        except ImportError:
            pass

    def log_attention_overlay(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        name: str = "attention",
    ) -> None:
        """Log attention map overlaid on input image."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb

            overlay = create_attention_overlay(image, attention_map)
            self._run.log(
                {f"attention/{name}": wandb.Image(overlay)}
            )
        except ImportError:
            pass

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    @property
    def offline_results(self) -> list[dict]:
        """Access results logged while W&B was unavailable."""
        return self._offline_results


def create_attention_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay attention heatmap on an image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        attention_map: (H, W) attention weights.
        alpha: Blending factor for overlay.

    Returns:
        (H, W, 3) uint8 image with attention overlay.
    """
    import matplotlib.cm as cm

    # Normalize attention to [0, 1]
    attn_min, attn_max = attention_map.min(), attention_map.max()
    if attn_max - attn_min > 1e-8:
        attn_norm = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attention_map)

    # Resize attention to image size if needed
    if attn_norm.shape != image.shape[:2]:
        from PIL import Image as PILImage

        attn_pil = PILImage.fromarray((attn_norm * 255).astype(np.uint8))
        attn_pil = attn_pil.resize(
            (image.shape[1], image.shape[0]), PILImage.BILINEAR
        )
        attn_norm = np.array(attn_pil).astype(np.float32) / 255.0

    # Apply colormap
    heatmap = (cm.jet(attn_norm)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (
        alpha * heatmap.astype(np.float32)
        + (1 - alpha) * image.astype(np.float32)
    ).astype(np.uint8)

    return overlay
