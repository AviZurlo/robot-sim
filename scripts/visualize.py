#!/usr/bin/env python
"""Generate visualization plots from training and evaluation data.

Reads loss_history.json from training and eval_metrics.json from evaluation,
produces clean dark-themed matplotlib plots saved as PNG files.

Usage:
    # Generate all plots from default paths
    python scripts/visualize.py

    # Specify paths explicitly
    python scripts/visualize.py \
        --train-log outputs/train/act_transfer_cube/loss_history.json \
        --eval-metrics outputs/eval/last/eval_metrics.json \
        --baseline-metrics outputs/eval/lerobot_act_aloha_sim_transfer_cube_human/eval_metrics.json \
        --output-dir outputs/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Dark theme colors
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
ACCENT_BLUE = "#4fc3f7"
ACCENT_ORANGE = "#ffb74d"
ACCENT_GREEN = "#81c784"
ACCENT_RED = "#ef5350"
ACCENT_PURPLE = "#ce93d8"


def setup_dark_style():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": PANEL_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.facecolor": PANEL_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.facecolor": BG_COLOR,
        "savefig.edgecolor": BG_COLOR,
    })


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_loss(loss_history: list[dict], output_dir: Path) -> Path:
    """Plot training loss curves (overall + components)."""
    steps = [h["step"] for h in loss_history]
    losses = [h["loss"] for h in loss_history]

    # Detect component losses
    component_keys = []
    for h in loss_history:
        for k in h:
            if k not in ("step", "loss") and k not in component_keys:
                component_keys.append(k)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw loss (faint) + smoothed loss (bold)
    ax.plot(steps, losses, color=ACCENT_BLUE, alpha=0.15, linewidth=0.5)
    if len(losses) > 20:
        smooth_steps = steps[19:]  # offset for moving average
        ax.plot(smooth_steps, smooth(losses), color=ACCENT_BLUE, linewidth=2.5, label="Total Loss")
    else:
        ax.plot(steps, losses, color=ACCENT_BLUE, linewidth=2.5, label="Total Loss")

    # Plot component losses if available
    colors = [ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]
    for i, key in enumerate(component_keys[:4]):
        vals = [h.get(key, float("nan")) for h in loss_history]
        valid = [(s, v) for s, v in zip(steps, vals) if not np.isnan(v)]
        if not valid:
            continue
        comp_steps, comp_vals = zip(*valid)
        color = colors[i % len(colors)]
        label = key.replace("_", " ").title()
        ax.plot(comp_steps, comp_vals, color=color, alpha=0.15, linewidth=0.5)
        if len(comp_vals) > 20:
            ax.plot(list(comp_steps)[19:], smooth(list(comp_vals)),
                    color=color, linewidth=2, label=label)
        else:
            ax.plot(comp_steps, comp_vals, color=color, linewidth=2, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("ACT Policy Training Loss", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    # Add annotations for start/end
    if losses:
        ax.annotate(f"Start: {losses[0]:.1f}", xy=(steps[0], losses[0]),
                     xytext=(steps[0] + len(steps) * 0.05, losses[0]),
                     color=TEXT_COLOR, fontsize=10, alpha=0.7)
        ax.annotate(f"End: {losses[-1]:.4f}", xy=(steps[-1], losses[-1]),
                     xytext=(steps[-1] - len(steps) * 0.3, losses[-1] + max(losses) * 0.05),
                     color=ACCENT_GREEN, fontsize=11, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, alpha=0.7))

    fig.tight_layout()
    path = output_dir / "training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_eval_success(eval_data: dict, output_dir: Path) -> Path:
    """Plot evaluation success rate across episodes."""
    episodes = eval_data["episodes"]
    summary = eval_data["summary"]

    n = len(episodes)
    successes = [1 if ep["success"] else 0 for ep in episodes]
    ep_labels = [f"Ep {i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = [ACCENT_GREEN if s else ACCENT_RED for s in successes]
    bars = ax.bar(ep_labels, successes, color=bar_colors, width=0.6, edgecolor="none")

    # Add success/fail labels on bars
    for bar, s in zip(bars, successes):
        label = "Pass" if s else "Fail"
        y = bar.get_height() + 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y, label,
                ha="center", va="bottom", fontsize=10,
                color=ACCENT_GREEN if s else ACCENT_RED, fontweight="bold")

    # Success rate line
    rate = summary["success_rate"]
    ax.axhline(y=rate, color=ACCENT_BLUE, linestyle="--", linewidth=2, alpha=0.8)
    label_y = max(rate + 0.08, 0.5)
    ax.text(n - 0.5, label_y, f"Success Rate: {rate*100:.0f}%",
            color=ACCENT_BLUE, fontsize=12, fontweight="bold", ha="right")

    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Success"])
    ax.set_title(f"Evaluation Results — {summary.get('checkpoint', 'Unknown')}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")

    fig.tight_layout()
    path = output_dir / "eval_success.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_eval_rewards(eval_data: dict, output_dir: Path) -> Path:
    """Plot reward per episode bar chart."""
    episodes = eval_data["episodes"]
    summary = eval_data["summary"]

    rewards = [ep["total_reward"] for ep in episodes]
    n = len(rewards)
    ep_labels = [f"Ep {i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color by success
    bar_colors = [ACCENT_GREEN if ep["success"] else ACCENT_ORANGE for ep in episodes]
    bars = ax.bar(ep_labels, rewards, color=bar_colors, width=0.6, edgecolor="none", alpha=0.9)

    # Add reward labels on bars
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{r:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)

    # Average line
    avg = summary["avg_reward"]
    max_reward = max(rewards) if rewards else 1
    y_top = max(max_reward * 1.2, 1.0)
    ax.axhline(y=avg, color=ACCENT_BLUE, linestyle="--", linewidth=2, alpha=0.8)
    ax.text(n - 0.5, avg + y_top * 0.05, f"Avg: {avg:.2f}",
            color=ACCENT_BLUE, fontsize=12, fontweight="bold", ha="right")

    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel("Total Reward")
    ax.set_xlabel("Episode")
    ax.set_title("Reward Per Episode", fontsize=14, fontweight="bold")

    fig.tight_layout()
    path = output_dir / "eval_rewards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_comparison(trained_data: dict, baseline_data: dict, output_dir: Path) -> Path:
    """Plot comparison between trained model and pretrained baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Data
    trained_sr = trained_data["summary"]["success_rate"] * 100
    baseline_sr = baseline_data["summary"]["success_rate"] * 100
    trained_reward = trained_data["summary"]["avg_reward"]
    baseline_reward = baseline_data["summary"]["avg_reward"]

    labels = ["Ours\n(500 steps)", "Pretrained\n(100k steps)"]

    # Success rate comparison
    bars1 = ax1.bar(labels, [trained_sr, baseline_sr],
                    color=[ACCENT_BLUE, ACCENT_ORANGE], width=0.5,
                    edgecolor="none", alpha=0.9)
    for bar, val in zip(bars1, [trained_sr, baseline_sr]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate Comparison", fontsize=14, fontweight="bold")

    # Reward comparison
    bars2 = ax2.bar(labels, [trained_reward, baseline_reward],
                    color=[ACCENT_BLUE, ACCENT_ORANGE], width=0.5,
                    edgecolor="none", alpha=0.9)
    for bar, val in zip(bars2, [trained_reward, baseline_reward]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Average Reward Comparison", fontsize=14, fontweight="bold")

    fig.suptitle("Trained vs Pretrained Baseline", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate training & evaluation plots")
    parser.add_argument("--train-log", type=str,
                        default="outputs/train/act_transfer_cube/loss_history.json",
                        help="Path to training loss_history.json")
    parser.add_argument("--eval-metrics", type=str,
                        default="outputs/eval/last/eval_metrics.json",
                        help="Path to evaluation eval_metrics.json")
    parser.add_argument("--baseline-metrics", type=str,
                        default="outputs/eval/lerobot_act_aloha_sim_transfer_cube_human/eval_metrics.json",
                        help="Path to pretrained baseline eval_metrics.json")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                        help="Directory to save plot PNGs")
    args = parser.parse_args()

    setup_dark_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    print(f"  Output: {output_dir}")
    print()

    plots = []

    # 1. Training loss curves
    train_log = Path(args.train_log)
    if train_log.exists():
        print("Training loss curves:")
        with open(train_log) as f:
            loss_history = json.load(f)
        plots.append(plot_training_loss(loss_history, output_dir))
    else:
        print(f"  Skipping training loss (not found: {train_log})")

    # 2. Evaluation success + rewards
    eval_path = Path(args.eval_metrics)
    if eval_path.exists():
        print("Evaluation plots:")
        with open(eval_path) as f:
            eval_data = json.load(f)
        plots.append(plot_eval_success(eval_data, output_dir))
        plots.append(plot_eval_rewards(eval_data, output_dir))
    else:
        print(f"  Skipping evaluation plots (not found: {eval_path})")

    # 3. Comparison chart
    baseline_path = Path(args.baseline_metrics)
    if eval_path.exists() and baseline_path.exists():
        print("Comparison chart:")
        with open(eval_path) as f:
            eval_data = json.load(f)
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        plots.append(plot_comparison(eval_data, baseline_data, output_dir))
    else:
        missing = []
        if not eval_path.exists():
            missing.append(str(eval_path))
        if not baseline_path.exists():
            missing.append(str(baseline_path))
        print(f"  Skipping comparison (not found: {', '.join(missing)})")

    print()
    if plots:
        print(f"Generated {len(plots)} plot(s) in {output_dir.resolve()}")
    else:
        print("No plots generated — run training and evaluation first.")


if __name__ == "__main__":
    main()
