"""Generate results charts and README section from probe JSON files.

Usage:
    python scripts/generate_results.py
    python scripts/generate_results.py --results-dir results/probes --figures-dir results/figures
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Model display config ─────────────────────────────────────────────
MODELS = {
    "xvla_widowx":          {"label": "X-VLA",          "scene": "WidowX", "color": "#4C72B0", "valid": True},
    "openvla_widowx":       {"label": "OpenVLA",         "scene": "WidowX", "color": "#DD8452", "valid": True},
    "pi0_widowx":           {"label": "Pi0†",            "scene": "WidowX", "color": "#aaaaaa", "valid": False},
    "pi0_franka":           {"label": "Pi0",             "scene": "Franka", "color": "#55A868", "valid": True},
    "openvla_oft_franka":   {"label": "OpenVLA-OFT",     "scene": "Franka", "color": "#C44E52", "valid": True},
    "cosmos_policy_franka": {"label": "Cosmos Policy",   "scene": "Franka", "color": "#8172B2", "valid": True},
}

# ── Key metrics per probe ─────────────────────────────────────────────
PROBE_METRICS = {
    "baseline": {
        "title": "Baseline",
        "metrics": [
            ("direction_alignment",  "Direction Alignment ↑", True),
            ("distance_to_target",   "Distance to Target ↓",  False),
            ("trajectory_spread",    "Trajectory Spread ↓",   False),
        ],
    },
    "spatial_symmetry": {
        "title": "Spatial Symmetry",
        "metrics": [
            ("swap_sensitivity",                          "Swap Sensitivity ↑",         True),
            ("swapped_distance_to_original_blue_pos",    "Endpoint→New Block Pos ↓",   False),
        ],
    },
    "camera_sensitivity": {
        "title": "Camera Sensitivity",
        "metrics": [
            ("mirror_camera_sensitivity", "Mirror Sensitivity ↑", True),
            ("flip_image_sensitivity",    "Flip Sensitivity ↑",   True),
        ],
    },
    "view_ablation": {
        "title": "View Ablation",
        "metrics": [
            ("primary_ablation_sensitivity",      "Primary Ablated ↑",   True),
            ("secondary_ablation_sensitivity",    "Secondary Ablated ↑", True),
            ("full_vision_ablation_sensitivity",  "Fully Blind ↑",       True),
        ],
    },
    "counterfactual": {
        "title": "Counterfactual",
        "metrics": [
            ("mean_synonym_sensitivity", "Synonym Sensitivity ↓", False),
            ("max_synonym_sensitivity",  "Max Synonym Sens. ↓",   False),
        ],
    },
    "null_action": {
        "title": "Null Action",
        "metrics": [
            ("mean_null_displacement",   "Null Displacement ↓",  False),
            ("null_vs_baseline_ratio",   "Null/Baseline Ratio ↓", False),
        ],
    },
    "perturbation": {
        "title": "Perturbation",
        "metrics": [
            ("mean_perturbation_sensitivity",       "Mean Sensitivity ↑",    True),
            ("sensitivity_displacement_correlation","Displacement Corr. ↑",  True),
        ],
    },
    "attention": {
        "title": "Attention",
        "metrics": [
            ("mean_attention_iou", "Mean IoU ↑", True),
        ],
    },
}

# Radar chart axes (normalized so outward = better)
RADAR_AXES = [
    ("baseline",           "direction_alignment",               True,  "Baseline\nAlignment"),
    ("spatial_symmetry",   "swap_sensitivity",                  True,  "Spatial\nGrounding"),
    ("camera_sensitivity", "mirror_camera_sensitivity",         True,  "Camera\nSensitivity"),
    ("view_ablation",      "primary_ablation_sensitivity",      True,  "Vision\nReliance"),
    ("perturbation",       "mean_perturbation_sensitivity",     True,  "Perturbation\nResponse"),
    ("null_action",        "null_vs_baseline_ratio",            False, "Null Action\nCompliance"),
    ("counterfactual",     "mean_synonym_sensitivity",          False, "Language\nRobustness"),
    ("attention",          "mean_attention_iou",                True,  "Attention\nAccuracy"),
]


def load_results(results_dir: Path) -> dict:
    data = {}
    for f in sorted(results_dir.glob("probe_results_*.json")):
        key = f.stem.replace("probe_results_", "")
        data[key] = json.loads(f.read_text())
    return data


def get_metric(data: dict, model: str, probe: str, metric: str):
    try:
        return data[model][probe]["metrics"][metric]
    except (KeyError, TypeError):
        return None


def normalize(values: list, higher_better: bool) -> list:
    """Normalize to [0,1] where 1 = best, handling None."""
    clean = [v for v in values if v is not None]
    if not clean or max(clean) == min(clean):
        return [0.5 if v is not None else 0.0 for v in values]
    lo, hi = min(clean), max(clean)
    normed = []
    for v in values:
        if v is None:
            normed.append(0.0)
        else:
            n = (v - lo) / (hi - lo)
            normed.append(n if higher_better else 1 - n)
    return normed


def make_radar_chart(data: dict, figures_dir: Path):
    N = len(RADAR_AXES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Collect and normalize each axis
    axis_values = {}
    for probe, metric, higher_better, label in RADAR_AXES:
        raw = [get_metric(data, m, probe, metric) for m in MODELS]
        axis_values[(probe, metric)] = normalize(raw, higher_better)

    for i, (model_key, cfg) in enumerate(MODELS.items()):
        vals = [axis_values[(p, m)][i] for p, m, _, _ in RADAR_AXES]
        vals += vals[:1]
        style = "--" if not cfg["valid"] else "-"
        alpha = 0.5 if not cfg["valid"] else 0.85
        ax.plot(angles, vals, style, color=cfg["color"], linewidth=2, alpha=alpha, label=cfg["label"])
        ax.fill(angles, vals, color=cfg["color"], alpha=0.08 if not cfg["valid"] else 0.12)

    labels = [label for _, _, _, label in RADAR_AXES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, color="white")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=7, color="#888")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#444")
    ax.grid(color="#444", linewidth=0.5)

    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.15),
        framealpha=0.3, facecolor="#1a1a2e", edgecolor="#444",
        labelcolor="white", fontsize=10,
    )

    ax.set_title("VLA Model Comparison — Diagnostic Probes", color="white", size=13, pad=20)
    fig.text(0.5, 0.02, "† Pi0 (WidowX) is out-of-distribution — results not valid for comparison",
             ha="center", color="#888", size=8)

    out = figures_dir / "radar_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


def make_probe_bar_charts(data: dict, figures_dir: Path):
    model_keys = list(MODELS.keys())
    colors = [MODELS[m]["color"] for m in model_keys]
    labels = [MODELS[m]["label"] for m in model_keys]

    for probe_key, probe_cfg in PROBE_METRICS.items():
        metrics = probe_cfg["metrics"]
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.5))
        fig.patch.set_facecolor("#0d1117")
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(probe_cfg["title"], color="white", size=13, y=1.02)

        for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics):
            ax.set_facecolor("#0d1117")
            vals = [get_metric(data, m, probe_key, metric_key) for m in model_keys]

            bar_colors = []
            for i, (m, v) in enumerate(zip(model_keys, vals)):
                c = MODELS[m]["color"]
                bar_colors.append(c if v is not None else "#333")

            bar_vals = [v if v is not None else 0 for v in vals]
            bars = ax.bar(labels, bar_vals, color=bar_colors, alpha=0.85, edgecolor="#333", linewidth=0.5)

            # Annotate N/A
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if v is None:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.01, "N/A",
                            ha="center", va="bottom", color="#888", fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + max(bar_vals) * 0.02,
                            f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=7.5)

            ax.set_title(metric_label, color="#ccc", size=10)
            ax.set_xticklabels(labels, rotation=30, ha="right", color="white", size=8)
            ax.tick_params(colors="#888")
            ax.spines[:].set_color("#444")
            ax.set_ylim(0, max(bar_vals) * 1.2 + 1e-6)
            for spine in ax.spines.values():
                spine.set_color("#444")
            ax.yaxis.label.set_color("#888")
            ax.tick_params(axis="y", colors="#888")

        plt.tight_layout()
        out = figures_dir / f"probe_{probe_key}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {out}")


def build_markdown_table(data: dict) -> str:
    model_keys = list(MODELS.keys())
    headers = ["Metric"] + [MODELS[m]["label"] + f"<br>({MODELS[m]['scene']})" for m in model_keys]
    sep = [":---"] + [":---:"] * len(model_keys)

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.3f}"

    rows = []
    for probe_key, probe_cfg in PROBE_METRICS.items():
        rows.append([f"**{probe_cfg['title']}**"] + [""] * len(model_keys))
        for metric_key, metric_label, _ in probe_cfg["metrics"]:
            row = [metric_label]
            for m in model_keys:
                row.append(fmt(get_metric(data, m, probe_key, metric_key)))
            rows.append(row)

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_results_section(data: dict, figures_rel: str = "results/figures") -> str:
    table = build_markdown_table(data)
    model_list = "\n".join(
        f"| **{cfg['label']}** | {cfg['scene']} | {'✓' if cfg['valid'] else '† OOD'} |"
        for cfg in MODELS.values()
    )

    return textwrap.dedent(f"""\
    ## VLA Diagnostic Probing

    Comparing 6 vision-language-action (VLA) models across 8 diagnostic probes in MuJoCo simulation.
    Each probe tests a specific aspect of visual and language grounding — from basic reaching to null-action
    compliance and attention localization.

    ### Models

    | Model | Scene | Valid |
    | :--- | :---: | :---: |
    {model_list}

    † Pi0 on WidowX is out-of-distribution (Pi0 was trained on Franka). Results are not valid for comparison.

    ### Radar Summary

    ![Radar comparison]({figures_rel}/radar_comparison.png)

    Each axis is normalized across all models (outward = better). See full metrics table below.

    ### Full Results

    {table}

    ↑ higher is better · ↓ lower is better · N/A = metric not available for this model/scene

    ### Probe Charts

    | | |
    |:---:|:---:|
    | ![Baseline]({figures_rel}/probe_baseline.png) | ![Spatial Symmetry]({figures_rel}/probe_spatial_symmetry.png) |
    | ![Camera Sensitivity]({figures_rel}/probe_camera_sensitivity.png) | ![View Ablation]({figures_rel}/probe_view_ablation.png) |
    | ![Counterfactual]({figures_rel}/probe_counterfactual.png) | ![Null Action]({figures_rel}/probe_null_action.png) |
    | ![Perturbation]({figures_rel}/probe_perturbation.png) | ![Attention]({figures_rel}/probe_attention.png) |

    ### Key Findings

    - **X-VLA** is the only model with genuine spatial grounding: it re-routes when blocks are swapped
      (swap\\_sensitivity 1.50) and responds to block movement (perturbation sensitivity 0.98). However its
      perturbation response is non-linear — it does not scale proportionally with displacement (correlation 0.24).
    - **OpenVLA** runs a near-fixed motor program: zero perturbation sensitivity, identical outputs for all
      synonym phrasings, and null/baseline ratio of exactly 1.0.
    - **Pi0 (Franka)** is highly stochastic (trajectory spread 0.44), making it difficult to isolate genuine
      scene conditioning from noise. Its perturbation correlation is strongly negative (-0.77), suggesting
      out-of-distribution collapse for larger block displacements.
    - **No model passes the null action test.** All models produce significant motion when instructed to
      stay still, with null/baseline ratios ranging from 0.82 (X-VLA, best) to 1.06 (Cosmos Policy, worst).
    - **Attention IoU is near zero for all models**, suggesting that spatial attention in these VLMs does
      not localize to task-relevant objects in a pixel-precise way.

    ### Reproducing Results

    ```bash
    # Run all probes for a model
    python -m vla_probing.run_all --model xvla --scene widowx --device mps

    # Regenerate this section and figures
    python scripts/generate_results.py
    ```
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/probes")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_results(results_dir)
    print(f"  Found: {list(data.keys())}")

    print("Generating radar chart...")
    make_radar_chart(data, figures_dir)

    print("Generating probe bar charts...")
    make_probe_bar_charts(data, figures_dir)

    print("Building README results section...")
    section = build_results_section(data)

    readme = Path(args.readme)
    content = readme.read_text()
    start_tag = "<!-- RESULTS_START -->"
    end_tag = "<!-- RESULTS_END -->"

    if start_tag in content and end_tag in content:
        new_content = re.sub(
            rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}",
            f"{start_tag}\n{section}\n{end_tag}",
            content,
            flags=re.DOTALL,
        )
    else:
        new_content = content + f"\n{start_tag}\n{section}\n{end_tag}\n"

    readme.write_text(new_content)
    print(f"  README updated: {readme}")
    print("Done.")


if __name__ == "__main__":
    main()
