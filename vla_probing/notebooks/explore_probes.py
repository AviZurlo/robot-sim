"""Interactive probe exploration script.

Run with: uv run python vla_probing/notebooks/explore_probes.py --model xvla --probe baseline

Renders the scene, runs a single probe, and saves visualizations to outputs/probes/viz/
"""

import argparse
from pathlib import Path

import numpy as np

# Output directory for visualizations
VIZ_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "probes" / "viz"


def render_scene_views(scene, title: str = "Scene"):
    """Save rendered camera views as images."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    views = scene.render_all_views()
    for name, img in views.items():
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        path = VIZ_DIR / f"scene_{name}.png"
        pil_img.save(path)
        print(f"  Saved: {path}")
    return views


def plot_trajectory(traj_xyz, target_pos=None, title="Trajectory", filename="trajectory.png"):
    """Plot 3D trajectory and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2],
            "b-o", markersize=3, linewidth=1.5, label="Predicted trajectory")
    ax.scatter(*traj_xyz[0], color="green", s=100, zorder=5, label="Start")
    ax.scatter(*traj_xyz[-1], color="red", s=100, zorder=5, label="End")

    if target_pos is not None:
        ax.scatter(*target_pos, color="orange", s=150, marker="*", zorder=5, label="Target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_attention_overlay(image, attention_map, title="Attention", filename="attention.png"):
    """Overlay attention heatmap on scene image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Scene")
    axes[0].axis("off")

    # Normalize attention
    attn = attention_map.copy()
    if attn.max() > attn.min():
        attn = (attn - attn.min()) / (attn.max() - attn.min())

    axes[1].imshow(attn, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(attn, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def run_baseline(adapter, scene):
    """Run baseline probe with full visualization."""
    from vla_probing.adapter import VLAInput

    print("\n📍 Running Baseline Probe...")
    print("  Prompt: 'pick up the red block'")

    views = render_scene_views(scene, "Default Scene")

    ee = scene.get_ee_state()
    inp = VLAInput(images=[views["image"], views["image2"]], prompt="pick up the red block", proprio=ee)

    adapter.reset()
    output = adapter.predict_action(inp)
    actions = np.atleast_2d(output.actions)
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    traj_xyz = actions[:, :3]

    red_pos = scene.get_block_pos("red")
    print(f"\n  Action shape: {output.actions.shape}")
    print(f"  First action: {output.actions[0][:7]}")
    print(f"  XYZ trajectory range: {traj_xyz.min(axis=0)} → {traj_xyz.max(axis=0)}")
    if red_pos is not None:
        print(f"  Red block position: {red_pos}")
        print(f"  Distance to target: {np.linalg.norm(traj_xyz[-1] - red_pos):.4f}")

    plot_trajectory(traj_xyz, target_pos=red_pos, title=f"Baseline — {adapter.model_name}", filename=f"baseline_{adapter.model_name}.png")

    # Attention
    print("\n  Extracting attention maps...")
    try:
        attn = adapter.get_attention(inp)
        if "spatial_attention" in attn:
            plot_attention_overlay(
                views["image"], attn["spatial_attention"],
                title=f"Attention — {adapter.model_name}",
                filename=f"attention_{adapter.model_name}.png",
            )
    except Exception as e:
        print(f"  Attention extraction failed: {e}")

    # Multi-seed stochasticity
    print("\n  Testing stochasticity (5 seeds)...")
    import torch
    spreads = []
    for seed in range(5):
        torch.manual_seed(seed)
        adapter.reset()
        out_s = adapter.predict_action(inp)
        a = np.atleast_2d(out_s.actions).reshape(-1, out_s.actions.shape[-1])[:, :3]
        spreads.append(a)
        print(f"    Seed {seed}: first XYZ = {a[0]}")

    from vla_probing.metrics import trajectory_spread
    spread = trajectory_spread(spreads)
    print(f"\n  Trajectory spread across seeds: {spread:.6f}")

    return output


def run_null_action(adapter, scene):
    """Run null action probe with visualization."""
    from vla_probing.adapter import VLAInput

    print("\n⛔ Running Null Action Probe...")

    views = scene.render_all_views()
    ee = scene.get_ee_state()

    prompts = ["pick up the red block", "don't move", "stay still", "do nothing"]

    for prompt in prompts:
        inp = VLAInput(images=[views["image"], views["image2"]], prompt=prompt, proprio=ee)
        adapter.reset()
        output = adapter.predict_action(inp)
        actions = np.atleast_2d(output.actions).reshape(-1, output.actions.shape[-1])
        displacement = np.linalg.norm(actions[-1, :3] - actions[0, :3]) if len(actions) > 1 else np.linalg.norm(actions[0, :3])
        print(f"  '{prompt}': action={output.actions[0][:7]}, displacement={displacement:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Interactive VLA probe exploration")
    parser.add_argument("--model", default="xvla", choices=["xvla", "pi0", "openvla"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--probe", default="baseline", choices=["baseline", "null_action", "all"])
    args = parser.parse_args()

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔬 VLA Probe Explorer — {args.model} on {args.device}")
    print(f"   Visualizations saved to: {VIZ_DIR}\n")

    from vla_probing.probes.base import make_adapter
    from vla_probing.scene import WidowXScene

    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()

    if args.probe in ("baseline", "all"):
        run_baseline(adapter, scene)

    if args.probe in ("null_action", "all"):
        run_null_action(adapter, scene)

    scene.close()
    print(f"\n✅ Done! Check {VIZ_DIR} for visualizations.")


if __name__ == "__main__":
    main()
