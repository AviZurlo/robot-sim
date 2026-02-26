#!/usr/bin/env python
"""Streamlit dashboard for VLA probing results.

Visualizes and compares diagnostic probe results across VLA models.
Reads JSON files from outputs/probes/probe_results_<model>.json.

Launch with:
    streamlit run vla_probing/dashboard.py --server.address 0.0.0.0 --server.port 8502

Or use the helper script:
    bash scripts/start_probe_dashboard.sh
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "outputs" / "probes"
ATTENTION_DIR = PROBES_DIR / "attention_maps"

# ---------------------------------------------------------------------------
# Probe metadata — descriptions of what each probe tests
# ---------------------------------------------------------------------------
PROBE_INFO = {
    "baseline": {
        "display": "Baseline",
        "description": "Baseline trajectory for 'pick up the red block'",
        "key_metric": "direction_alignment",
        "metric_explanation": "Does the arm move toward the target? 1.0 = perfect, 0 = random, negative = moves away.",
        "higher_is_better": True,
        "tests": "Basic pick-and-place competence and trajectory quality.",
    },
    "spatial_symmetry": {
        "display": "Spatial Symmetry",
        "description": "Swap block positions to test spatial understanding",
        "key_metric": "perturbation_sensitivity",
        "metric_explanation": "How much does the trajectory change when block positions are swapped? Higher = more spatially aware.",
        "higher_is_better": True,
        "tests": "Whether the model adapts its trajectory when object positions are swapped.",
    },
    "camera_sensitivity": {
        "display": "Camera Sensitivity",
        "description": "Mirror/rotate camera to test spatial understanding",
        "key_metric": "mirror_camera_sensitivity",
        "metric_explanation": "How much does mirroring the camera change the output? Higher = more sensitive to camera view.",
        "higher_is_better": True,
        "tests": "Robustness to camera transformations (mirror, flip).",
    },
    "view_ablation": {
        "display": "View Ablation",
        "description": "Remove primary/secondary camera views",
        "key_metric": "full_vision_ablation_sensitivity",
        "metric_explanation": "How much does blacking out all cameras change output? 0 = ignores vision entirely (bad).",
        "higher_is_better": True,
        "tests": "Dependence on each camera view — which views carry the most information.",
    },
    "counterfactual": {
        "display": "Counterfactual",
        "description": "Test language understanding with synonym variations",
        "key_metric": "mean_synonym_sensitivity",
        "metric_explanation": "Do synonyms ('grab' vs 'pick up') produce different actions? Low = good semantic understanding.",
        "higher_is_better": False,
        "tests": "Language grounding — do synonyms produce similar actions?",
    },
    "null_action": {
        "display": "Null Action",
        "description": "Test null action compliance with 'don't move' prompts",
        "key_metric": "null_vs_baseline_ratio",
        "metric_explanation": "Movement when told 'don't move' ÷ normal movement. 0 = stays still (good), 1.0 = ignores instruction (bad).",
        "higher_is_better": False,
        "tests": "Whether the model can stay still when told not to move.",
    },
    "attention": {
        "display": "Attention",
        "description": "Extract and visualize attention maps",
        "key_metric": "mean_attention_iou",
        "metric_explanation": "Does attention overlap with the referenced object? 1.0 = focused on target, 0 = looking elsewhere.",
        "higher_is_better": True,
        "tests": "Whether attention focuses on the referenced object.",
    },
    "perturbation": {
        "display": "Perturbation",
        "description": "Move blocks to test trajectory adaptation",
        "key_metric": "mean_perturbation_sensitivity",
        "metric_explanation": "When a block is shifted, does the trajectory adapt? Higher = more responsive to scene changes.",
        "higher_is_better": True,
        "tests": "Sensitivity to object position changes — does the model re-plan?",
    },
}

# Consistent color palette for models (base model -> color)
MODEL_COLORS = {
    "xvla": "#636EFA",
    "pi0": "#EF553B",
    "smolvla": "#00CC96",
    "openvla": "#AB63FA",
}

# Scene suffixes get a lighter shade
SCENE_COLOR_VARIANTS = {
    "franka": 0,     # primary shade
    "widowx": 1,     # secondary shade
}

MODEL_DISPLAY = {
    "xvla": "X-VLA",
    "pi0": "Pi0",
    "smolvla": "SmolVLA",
    "openvla": "OpenVLA",
}

SCENE_DISPLAY = {
    "franka": "Franka",
    "widowx": "WidowX",
}

# Known scene suffixes for parsing result filenames
KNOWN_SCENES = {"franka", "widowx"}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VLA Probing Dashboard",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_result_key(stem: str) -> tuple[str, str, str]:
    """Parse a result filename stem into (key, base_model, scene).

    Examples:
        probe_results_pi0_franka -> ("pi0_franka", "pi0", "franka")
        probe_results_pi0        -> ("pi0", "pi0", "")
        probe_results_xvla       -> ("xvla", "xvla", "")
    """
    name = stem.replace("probe_results_", "")
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in KNOWN_SCENES:
        return name, parts[0], parts[1]
    return name, name, ""


@st.cache_data(ttl=30)
def load_all_results() -> dict[str, dict]:
    """Load probe results for all available models/scenes."""
    results = {}
    if not PROBES_DIR.exists():
        return results
    for f in sorted(PROBES_DIR.glob("probe_results_*.json")):
        key, _, _ = _parse_result_key(f.stem)
        with open(f) as fp:
            results[key] = json.load(fp)
    return results


def get_model_meta(data: dict) -> dict:
    """Extract _meta block from model results."""
    return data.get("_meta", {})


def get_model_color(key: str) -> str:
    """Get color for a model key (e.g. 'pi0_franka' -> pi0 color)."""
    _, base, _ = _parse_result_key("probe_results_" + key)
    return MODEL_COLORS.get(base, "#888888")


def get_model_display(key: str) -> str:
    """Get display name (e.g. 'pi0_franka' -> 'Pi0 (Franka)')."""
    _, base, scene = _parse_result_key("probe_results_" + key)
    base_display = MODEL_DISPLAY.get(base, base.upper())
    if scene:
        scene_display = SCENE_DISPLAY.get(scene, scene.title())
        return f"{base_display} ({scene_display})"
    return base_display


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("VLA Probing")

all_results = load_all_results()
model_names = [m for m in all_results if m != "_meta"]

if model_names:
    st.sidebar.caption(f"{len(model_names)} model(s) with results")
else:
    st.sidebar.caption("No probe results found")

page = st.sidebar.radio(
    "Navigate",
    ["About", "Overview", "Per-Probe Details", "Metrics Explorer", "Attention Maps"],
)

st.sidebar.divider()

if model_names:
    selected_models = st.sidebar.multiselect(
        "Models to compare",
        model_names,
        default=model_names,
        format_func=get_model_display,
    )
else:
    selected_models = []

st.sidebar.divider()
st.sidebar.caption(
    "Results from `outputs/probes/probe_results_<model>.json`\n\n"
    "Run probes: `python -m vla_probing --model <name>`"
)


# ---------------------------------------------------------------------------
# No data state
# ---------------------------------------------------------------------------
if not model_names:
    st.header("VLA Probing Dashboard")
    st.info(
        "No probe results found. Run the probing suite to generate results:\n\n"
        "```bash\n"
        "# Run all probes for X-VLA\n"
        "python -m vla_probing --model xvla --device mps\n\n"
        "# Results will be saved to outputs/probes/probe_results_xvla.json\n"
        "```\n\n"
        "Expected location: `outputs/probes/probe_results_<model>.json`"
    )
    st.stop()


# ===================================================================
# PAGE: About
# ===================================================================
if page == "About":
    st.header("VLA Comparison Project")
    st.subheader("Debugging as Architecture Insight")

    st.markdown("""
This project runs **diagnostic probes** across multiple Vision-Language-Action (VLA)
models to understand what they actually learn about vision, language, and action —
and where they break.

### Inspiration

This work is inspired by [**Avik De's article: "Debugging as Architecture Insight:
Dissecting a VLA"**](https://www.avikde.me/p/debugging-as-architecture-insight),
which probes X-VLA to reveal that VLAs often learn spatial heuristics from training
data rather than genuine scene understanding, fail silently (producing plausible but
wrong trajectories), and can't be debugged like classical robotics systems.

We extend his approach by running the **same diagnostic experiments across multiple
VLA architectures** to compare them head-to-head.

### The 8 Diagnostic Probes

Each probe tests a specific hypothesis about what the model has learned:

| # | Probe | What It Tests |
|---|-------|--------------|
| 1 | **Baseline Trajectory** | Does the model reach for the correct object? |
| 2 | **Spatial Symmetry** | Does the model understand absolute vs. relative positions? |
| 3 | **Camera Sensitivity** | Is spatial reasoning tied to camera pose or truly 3D? |
| 4 | **View Ablation** | Which camera views carry the most information? |
| 5 | **Counterfactual Prompts** | Does the language encoder collapse synonyms correctly? |
| 6 | **Null Action** | Can the model comply with "don't move"? |
| 7 | **Attention Visualization** | Is the model attending to the referenced object? |
| 8 | **Environment Perturbation** | Does the model re-plan when objects move? |

### Models Under Comparison

| Model | Params | Architecture | Action Space | Embodiment |
|-------|--------|-------------|-------------|------------|
| **X-VLA** | 0.9B | InternVL2 + soft prompts + flow matching | Continuous | WidowX |
| **π0** | 3B | PaliGemma + flow matching | Continuous | LIBERO / cross-embodiment |
| **SmolVLA** | ~0.5B | SmolVLM2 + flow matching | Continuous | SO-100 / community |
| **OpenVLA** | 7B | Llama-2 + DINOv2 + SigLIP → text tokens | Discrete (text) | WidowX / BridgeV2 |

### Key Research Questions

1. Do different action representations (continuous flow matching vs. text tokens) produce fundamentally different failure modes?
2. Does model scale (0.5B → 7B) improve spatial understanding, or just task coverage?
3. Are soft prompts (X-VLA) encoding embodiment-specific spatial reasoning, or just biases?
4. Does any architecture show genuine null-action compliance?
5. How much of VLA behavior is spatial template matching vs. actual scene understanding?

### What's Novel

Nobody has done systematic **interpretability probing** across VLA architectures.
Existing comparisons (e.g., [RoboVLMs](https://robovlms.github.io/)) measure task
success rates. We measure *what the models understand*.

---

### 🎲 How We Measure Variance: Flow Matching vs. Autoregressive

These models generate actions in fundamentally different ways, which affects how we
measure their confidence and consistency.

#### Flow Matching (X-VLA, π0, SmolVLA)

These models start from **random noise** and iteratively "denoise" it into an action,
similar to how image diffusion models work. Think of it like starting with a page of
random scribbles and gradually cleaning them up into a drawing.

The **seed** controls which random noise you start with. Different seeds → different
starting scribbles → slightly different final actions. No seed is inherently "better" —
they're just different starting points. A robust model should produce similar actions
regardless of which seed is used.

We run each probe with **multiple seeds** (default: 10) to measure how much the model's
output varies. High variance means the model is uncertain; low variance means it's
confident in its prediction.

#### Autoregressive / Discrete Tokens (OpenVLA)

OpenVLA generates actions as **text tokens**, one at a time — like writing a sentence
word by word. At each step, it picks from a set of 256 discrete bins per action dimension.

The **sampling temperature** controls how "adventurous" the model is when picking tokens:
- **Temperature = 0**: Always picks the highest-probability token → fully deterministic,
  identical output every time
- **Temperature = 0.5**: Occasionally picks less-likely-but-reasonable tokens →
  some variance, good for measuring confidence
- **Temperature = 1.0**: Picks more freely → more variance, but may produce noisier actions

#### Our Setup

| Model | Variance Mechanism | Setting | Samples per Probe |
|-------|-------------------|---------|-------------------|
| X-VLA | Flow matching seeds | Seeds 0–9 | 10 |
| π0 | Flow matching seeds | Seeds 0–9 | 10 |
| SmolVLA | Flow matching seeds | Seeds 0–9 | 10 |
| OpenVLA | Sampling temperature | **T = 0.5** | 10 |

We chose **temperature 0.5** for OpenVLA as a balance: enough randomness to reveal
the model's uncertainty distribution, but not so much that actions become unreasonably
noisy. This is consistent with values used in the robotics literature (typically 0.3–0.7).

⚠️ **Note:** OpenVLA's 256-bin discretization also creates a "quantization dead zone" —
small input changes (camera shifts, prompt synonyms, minor position changes) may not
cross bin boundaries, producing identical discrete outputs even with temperature > 0.
This is a fundamental architectural property, not a bug.

---

**References:**
- [Avik De — Debugging as Architecture Insight](https://www.avikde.me/p/debugging-as-architecture-insight)
- [Avik De — The Architecture Behind "End-to-End" Robotics Pipelines](https://www.avikde.me/p/the-architecture-behind-end-to-end)
- [X-VLA Paper (arXiv 2510.10274)](https://arxiv.org/abs/2510.10274)
- [avikde/vla-pipeline (GitHub)](https://github.com/avikde/vla-pipeline)
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
""")

# ===================================================================
# PAGE: Overview
# ===================================================================
elif page == "Overview":
    st.header("Probe Results Overview")

    # ----- Model metadata cards -----
    st.subheader("Models")
    cols = st.columns(max(len(selected_models), 1))
    for i, model in enumerate(selected_models):
        meta = get_model_meta(all_results[model])
        with cols[i % len(cols)]:
            st.markdown(
                f"**{get_model_display(model)}**"
            )
            if meta:
                st.caption(
                    f"Architecture: {meta.get('architecture', '—')}\n\n"
                    f"Parameters: {meta.get('params_m', '?')}M\n\n"
                    f"Embodiment: {meta.get('embodiment', '—')}\n\n"
                    f"Device: {meta.get('device', '—')}\n\n"
                    f"Runtime: {meta.get('total_elapsed_s', '?'):.0f}s"
                    if isinstance(meta.get("total_elapsed_s"), (int, float))
                    else f"Runtime: {meta.get('total_elapsed_s', '?')}"
                )
            else:
                st.caption("No metadata available")

    # ----- Summary heatmap table: model x probe -> key metric -----
    st.subheader("Summary — Key Metric per Probe")

    probe_names = list(PROBE_INFO.keys())
    rows = []
    for model in selected_models:
        data = all_results[model]
        row = {"Model": get_model_display(model)}
        for probe in probe_names:
            probe_data = data.get(probe, {})
            if "error" in probe_data:
                row[PROBE_INFO[probe]["display"]] = None
            elif "metrics" in probe_data:
                key = PROBE_INFO[probe]["key_metric"]
                row[PROBE_INFO[probe]["display"]] = probe_data["metrics"].get(key)
            else:
                row[PROBE_INFO[probe]["display"]] = None
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Model")

    # Normalize per-probe so green always = good, red always = bad.
    # For "higher_is_better" probes, high raw value → high normalized (green).
    # For "lower_is_better" probes, low raw value → high normalized (green).
    import numpy as np

    norm_z = np.full_like(summary_df.values, np.nan, dtype=float)
    hover_texts = []
    for j, probe in enumerate(probe_names):
        col = summary_df.iloc[:, j].values.astype(float)
        info = PROBE_INFO[probe]
        higher_is_better = info.get("higher_is_better", True)
        col_min = np.nanmin(col) if not np.all(np.isnan(col)) else 0
        col_max = np.nanmax(col) if not np.all(np.isnan(col)) else 1
        col_range = col_max - col_min if col_max != col_min else 1.0
        for i in range(len(col)):
            if np.isnan(col[i]):
                norm_z[i, j] = np.nan
            else:
                scaled = (col[i] - col_min) / col_range
                norm_z[i, j] = scaled if higher_is_better else (1.0 - scaled)

    # Build display text (raw values) and hover text (with explanation)
    display_text = summary_df.map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
    ).values
    hover_text = []
    for i in range(len(summary_df)):
        row_hover = []
        for j, probe in enumerate(probe_names):
            info = PROBE_INFO[probe]
            val = summary_df.iloc[i, j]
            direction = "↑ higher is better" if info.get("higher_is_better", True) else "↓ lower is better"
            explanation = info.get("metric_explanation", "")
            val_str = f"{val:.4f}" if pd.notna(val) else "—"
            row_hover.append(
                f"<b>{info['display']}</b> ({info['key_metric']})<br>"
                f"Value: {val_str} ({direction})<br>"
                f"{explanation}"
            )
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=norm_z,
            x=summary_df.columns.tolist(),
            y=summary_df.index.tolist(),
            colorscale="RdYlGn",
            text=display_text,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverongaps=False,
            showscale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Score",
                tickvals=[0, 0.5, 1],
                ticktext=["Bad", "Mid", "Good"],
            ),
        )
    )
    fig.update_layout(
        height=max(200, 80 * len(selected_models)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend explaining each probe's key metric
    with st.expander("📖 How to read this chart"):
        st.markdown(
            "Each cell shows the **raw value** of the most important metric for that probe. "
            "Colors are normalized **per-probe** so that **green = good** and **red = bad**, "
            "regardless of whether higher or lower is better for that metric.\n\n"
            "Hover over any cell for a detailed explanation.\n"
        )
        legend_rows = []
        for probe in probe_names:
            info = PROBE_INFO[probe]
            direction = "↑ Higher is better" if info.get("higher_is_better", True) else "↓ Lower is better"
            legend_rows.append({
                "Probe": info["display"],
                "Key Metric": f"`{info['key_metric']}`",
                "Direction": direction,
                "What it measures": info.get("metric_explanation", ""),
            })
        st.table(pd.DataFrame(legend_rows).set_index("Probe"))

    # Methodology note
    with st.expander("🎲 Variance methodology"):
        st.markdown(
            "**Flow matching models** (X-VLA, π0, SmolVLA) are tested with 10 random seeds "
            "to measure output variance. Different seeds produce different random starting noise "
            "for the denoising process.\n\n"
            "**Autoregressive models** (OpenVLA) use **sampling temperature = 0.5** to introduce "
            "controlled randomness in token selection. Temperature 0 would be fully deterministic; "
            "0.5 balances variance measurement with action quality.\n\n"
            "See the **About** page for a full explanation of these mechanisms."
        )

    # ----- Awaiting results -----
    base_models_present = {_parse_result_key("probe_results_" + m)[1] for m in model_names}
    all_base_models = ["xvla", "pi0", "smolvla", "openvla"]
    missing = [m for m in all_base_models if m not in base_models_present]
    if missing:
        st.caption(
            "Awaiting results for: "
            + ", ".join(MODEL_DISPLAY.get(m, m.upper()) for m in missing)
        )

    # ----- Radar / spider chart comparing models -----
    st.subheader("Radar Comparison")

    # Normalize key metrics to [0, 1] for radar chart
    radar_data = {}
    for model in selected_models:
        data = all_results[model]
        values = []
        for probe in probe_names:
            probe_data = data.get(probe, {})
            if "metrics" in probe_data:
                key = PROBE_INFO[probe]["key_metric"]
                values.append(probe_data["metrics"].get(key, 0))
            else:
                values.append(0)
        radar_data[model] = values

    # Normalize each probe's values across models to [0, 1]
    categories = [PROBE_INFO[p]["display"] for p in probe_names]
    radar_fig = go.Figure()

    for model in selected_models:
        vals = radar_data[model]
        # Min-max normalize per probe across all selected models
        norm_vals = []
        for j in range(len(probe_names)):
            col_vals = [radar_data[m][j] for m in selected_models]
            cmin, cmax = min(col_vals), max(col_vals)
            if cmax - cmin > 1e-10:
                norm_vals.append((vals[j] - cmin) / (cmax - cmin))
            else:
                norm_vals.append(0.5)
        # Close the polygon
        norm_vals.append(norm_vals[0])

        radar_fig.add_trace(
            go.Scatterpolar(
                r=norm_vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=get_model_display(model),
                line=dict(color=get_model_color(model)),
                opacity=0.7,
            )
        )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ----- Execution time breakdown -----
    st.subheader("Probe Execution Time")

    time_rows = []
    for model in selected_models:
        data = all_results[model]
        for probe in probe_names:
            probe_data = data.get(probe, {})
            elapsed = probe_data.get("elapsed_s", 0)
            time_rows.append({
                "Model": get_model_display(model),
                "Probe": PROBE_INFO[probe]["display"],
                "Time (s)": elapsed,
            })

    if time_rows:
        time_df = pd.DataFrame(time_rows)
        fig = px.bar(
            time_df,
            x="Probe",
            y="Time (s)",
            color="Model",
            barmode="group",
            color_discrete_map={
                get_model_display(m): get_model_color(m) for m in selected_models
            },
        )
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: Per-Probe Details
# ===================================================================
elif page == "Per-Probe Details":
    st.header("Per-Probe Details")

    probe_options = {
        PROBE_INFO[p]["display"]: p for p in PROBE_INFO
    }
    selected_display = st.selectbox("Select Probe", list(probe_options.keys()))
    selected_probe = probe_options[selected_display]
    info = PROBE_INFO[selected_probe]

    st.markdown(f"**{info['description']}**")
    st.caption(f"Tests: {info['tests']}")
    st.divider()

    # Collect metrics for this probe across models
    probe_metrics = {}
    for model in selected_models:
        data = all_results[model]
        probe_data = data.get(selected_probe, {})
        if "error" in probe_data:
            st.warning(f"{get_model_display(model)}: probe failed — {probe_data['error']}")
        elif "metrics" in probe_data:
            probe_metrics[model] = probe_data["metrics"]

    if not probe_metrics:
        st.info("No results available for this probe with the selected models.")
        st.stop()

    # Side-by-side metric cards
    cols = st.columns(max(len(probe_metrics), 1))
    for i, (model, metrics) in enumerate(probe_metrics.items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{get_model_display(model)}**")
            key_metric = info["key_metric"]
            if key_metric in metrics:
                st.metric(
                    key_metric.replace("_", " ").title(),
                    f"{metrics[key_metric]:.6f}",
                )
            for k, v in sorted(metrics.items()):
                if k != key_metric:
                    st.caption(f"{k}: {v:.6f}")

    st.divider()

    # Bar chart comparing all metrics for this probe
    st.subheader("Metric Comparison")

    # Get union of all metric keys
    all_metric_keys = set()
    for metrics in probe_metrics.values():
        all_metric_keys.update(metrics.keys())
    all_metric_keys = sorted(all_metric_keys)

    bar_rows = []
    for model, metrics in probe_metrics.items():
        for k in all_metric_keys:
            bar_rows.append({
                "Model": get_model_display(model),
                "Metric": k.replace("_", " ").title(),
                "Value": metrics.get(k, 0),
            })

    if bar_rows:
        bar_df = pd.DataFrame(bar_rows)
        fig = px.bar(
            bar_df,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            color_discrete_map={
                get_model_display(m): get_model_color(m) for m in probe_metrics
            },
        )
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Variant and timing info
    st.subheader("Run Details")
    detail_rows = []
    for model in probe_metrics:
        probe_data = all_results[model].get(selected_probe, {})
        detail_rows.append({
            "Model": get_model_display(model),
            "Variant": probe_data.get("variant", "—"),
            "Time (s)": f"{probe_data.get('elapsed_s', 0):.1f}",
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Metrics Explorer
# ===================================================================
elif page == "Metrics Explorer":
    st.header("Metrics Explorer")

    # Flatten all metrics into a single dataframe
    flat_rows = []
    for model in selected_models:
        data = all_results[model]
        for probe_name, probe_data in data.items():
            if probe_name.startswith("_"):
                continue
            if "metrics" not in probe_data:
                continue
            for metric_name, value in probe_data["metrics"].items():
                flat_rows.append({
                    "model": get_model_display(model),
                    "model_key": model,
                    "probe": PROBE_INFO.get(probe_name, {}).get("display", probe_name),
                    "probe_key": probe_name,
                    "variant": probe_data.get("variant", ""),
                    "metric": metric_name,
                    "value": value,
                })

    if not flat_rows:
        st.info("No metrics data available.")
        st.stop()

    flat_df = pd.DataFrame(flat_rows)
    all_metrics = sorted(flat_df["metric"].unique())

    # ----- Scatter plot: metric vs metric -----
    st.subheader("Scatter: Metric vs Metric")

    col1, col2 = st.columns(2)
    with col1:
        x_metric = st.selectbox("X axis", all_metrics, index=0)
    with col2:
        y_default = min(1, len(all_metrics) - 1)
        y_metric = st.selectbox("Y axis", all_metrics, index=y_default)

    # Build scatter data — pivot so each row has both metrics
    scatter_rows = []
    for model in selected_models:
        data = all_results[model]
        for probe_name, probe_data in data.items():
            if probe_name.startswith("_") or "metrics" not in probe_data:
                continue
            metrics = probe_data["metrics"]
            if x_metric in metrics and y_metric in metrics:
                scatter_rows.append({
                    "Model": get_model_display(model),
                    "Probe": PROBE_INFO.get(probe_name, {}).get("display", probe_name),
                    x_metric: metrics[x_metric],
                    y_metric: metrics[y_metric],
                })

    if scatter_rows:
        scatter_df = pd.DataFrame(scatter_rows)
        fig = px.scatter(
            scatter_df,
            x=x_metric,
            y=y_metric,
            color="Model",
            symbol="Probe",
            hover_data=["Model", "Probe"],
            color_discrete_map={
                get_model_display(m): get_model_color(m) for m in selected_models
            },
        )
        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No probes have both selected metrics.")

    st.divider()

    # ----- Distribution: metric values across probes -----
    st.subheader("Distribution: Metric Across Probes")

    dist_metric = st.selectbox(
        "Metric to plot",
        all_metrics,
        index=all_metrics.index("trajectory_jerk") if "trajectory_jerk" in all_metrics else 0,
        key="dist_metric",
    )

    dist_df = flat_df[flat_df["metric"] == dist_metric]
    if not dist_df.empty:
        fig = px.box(
            dist_df,
            x="model",
            y="value",
            color="model",
            points="all",
            color_discrete_map={
                get_model_display(m): get_model_color(m) for m in selected_models
            },
            labels={"value": dist_metric, "model": "Model"},
        )
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- Filterable metric table -----
    st.subheader("Full Metrics Table")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_probes = st.multiselect(
            "Filter probes",
            flat_df["probe"].unique().tolist(),
            default=flat_df["probe"].unique().tolist(),
        )
    with filter_col2:
        filter_metrics = st.multiselect(
            "Filter metrics",
            all_metrics,
            default=all_metrics,
        )

    filtered = flat_df[
        (flat_df["probe"].isin(filter_probes))
        & (flat_df["metric"].isin(filter_metrics))
    ]

    # Pivot for display: rows = (probe, metric), cols = model
    if not filtered.empty:
        pivot = filtered.pivot_table(
            index=["probe", "metric"],
            columns="model",
            values="value",
            aggfunc="first",
        )
        st.dataframe(
            pivot.style.format("{:.6f}", na_rep="—"),
            use_container_width=True,
        )


# ===================================================================
# PAGE: Attention Maps
# ===================================================================
elif page == "Attention Maps":
    st.header("Attention Maps")

    if not ATTENTION_DIR.exists():
        st.info(
            "No attention map images found.\n\n"
            "Attention maps are saved when running the attention probe:\n\n"
            "```bash\n"
            "python -m vla_probing attention --model xvla\n"
            "```\n\n"
            f"Expected location: `{ATTENTION_DIR.relative_to(ROOT)}/`"
        )

        # Still show attention metrics if available
        has_attn_metrics = False
        for model in selected_models:
            data = all_results[model]
            if "attention" in data and "metrics" in data["attention"]:
                has_attn_metrics = True
                break

        if has_attn_metrics:
            st.subheader("Attention IoU Metrics")
            for model in selected_models:
                data = all_results[model]
                attn_data = data.get("attention", {})
                if "metrics" not in attn_data:
                    continue

                st.markdown(f"**{get_model_display(model)}**")
                metrics = attn_data["metrics"]

                # Extract IoU metrics
                iou_metrics = {
                    k: v for k, v in metrics.items() if k.startswith("iou_")
                }
                if iou_metrics:
                    iou_rows = []
                    for k, v in iou_metrics.items():
                        prompt = k.replace("iou_", "").replace("_", " ")
                        iou_rows.append({"Prompt": prompt, "IoU": v})

                    iou_df = pd.DataFrame(iou_rows)
                    fig = px.bar(
                        iou_df,
                        x="Prompt",
                        y="IoU",
                        color_discrete_sequence=[get_model_color(model)],
                    )
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(range=[0, 1]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                mean_iou = metrics.get("mean_attention_iou")
                if mean_iou is not None:
                    st.metric("Mean Attention IoU", f"{mean_iou:.3f}")
    else:
        # Display saved attention map images
        image_files = sorted(ATTENTION_DIR.glob("*.png"))

        if not image_files:
            st.info("Attention maps directory exists but contains no PNG images.")
        else:
            # Group by model
            model_images: dict[str, list[Path]] = {}
            for img in image_files:
                # Expected naming: <model>_<prompt>.png
                parts = img.stem.split("_", 1)
                model_key = parts[0] if len(parts) > 1 else "unknown"
                model_images.setdefault(model_key, []).append(img)

            # Side-by-side display
            if len(model_images) > 1:
                st.subheader("Side-by-Side Comparison")
                cols = st.columns(len(model_images))
                for i, (model_key, images) in enumerate(model_images.items()):
                    with cols[i]:
                        st.markdown(f"**{get_model_display(model_key)}**")
                        for img in images:
                            label = img.stem.split("_", 1)[-1].replace("_", " ")
                            st.caption(label)
                            st.image(str(img), use_container_width=True)
            else:
                # Single model
                for model_key, images in model_images.items():
                    st.subheader(f"{get_model_display(model_key)} Attention Maps")
                    cols = st.columns(min(len(images), 3))
                    for j, img in enumerate(images):
                        with cols[j % 3]:
                            label = img.stem.split("_", 1)[-1].replace("_", " ")
                            st.caption(label)
                            st.image(str(img), use_container_width=True)

        # Also show IoU metrics
        st.divider()
        st.subheader("Attention IoU Metrics")

        iou_bar_rows = []
        for model in selected_models:
            data = all_results[model]
            attn_data = data.get("attention", {})
            if "metrics" not in attn_data:
                continue
            for k, v in attn_data["metrics"].items():
                if k.startswith("iou_"):
                    iou_bar_rows.append({
                        "Model": get_model_display(model),
                        "Prompt": k.replace("iou_", "").replace("_", " "),
                        "IoU": v,
                    })

        if iou_bar_rows:
            iou_bar_df = pd.DataFrame(iou_bar_rows)
            fig = px.bar(
                iou_bar_df,
                x="Prompt",
                y="IoU",
                color="Model",
                barmode="group",
                color_discrete_map={
                    get_model_display(m): get_model_color(m) for m in selected_models
                },
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)
