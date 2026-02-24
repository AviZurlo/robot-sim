#!/usr/bin/env python
"""Live Streamlit dashboard for monitoring robot training experiments.

Launch with:
    streamlit run scripts/dashboard.py --server.address 0.0.0.0 --server.port 8501

Or use the helper script:
    bash scripts/start_dashboard.sh
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "outputs" / "train" / "act_transfer_cube"
EVAL_DIR = ROOT / "outputs" / "eval"
PLOTS_DIR = ROOT / "outputs" / "plots"
LOSS_LOG = TRAIN_DIR / "loss_history.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Robot Training Dashboard",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_eval_dirs() -> list[Path]:
    """Return eval directories sorted by modification time (newest first)."""
    if not EVAL_DIR.exists():
        return []
    dirs = [d for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "eval_metrics.json").exists()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def find_videos(directory: Path) -> list[Path]:
    """Find mp4 files in a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.mp4"))


def is_training_running() -> bool:
    """Check if a training process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "scripts/train.py"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def log_recently_updated(seconds: int = 30) -> bool:
    """Check if the loss log was updated within the last N seconds."""
    if not LOSS_LOG.exists():
        return False
    mtime = LOSS_LOG.stat().st_mtime
    return (time.time() - mtime) < seconds


def format_timestamp(path: Path) -> str:
    """Format a file's modification time."""
    mtime = path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Robot Training")
st.sidebar.caption("ACT Policy — ALOHA Sim")

page = st.sidebar.radio(
    "Navigate",
    ["Live Training", "Evaluation Results", "Experiment History",
     "Baseline Comparison", "Status"],
)

st.sidebar.divider()
refresh = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 5)
if refresh > 0:
    st.sidebar.caption(f"Refreshing every {refresh}s")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if refresh > 0:
    time.sleep(0.1)  # small delay to let page render
    st.empty()
    # Use st.fragment / auto_refresh via query param trick
    # Streamlit's st_autorefresh is in streamlit-autorefresh; we use native rerun
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= refresh:
        st.session_state.last_refresh = time.time()
        st.rerun()


# ===================================================================
# PAGE: Live Training
# ===================================================================
if page == "Live Training":
    st.header("Live Training Loss")

    loss_history = load_json(LOSS_LOG)

    if loss_history is None or len(loss_history) == 0:
        st.info(
            "No training data yet. Start training with:\n\n"
            "```bash\npython scripts/train.py --steps 5000 --device mps\n```"
        )
    else:
        # Status indicator
        if is_training_running() or log_recently_updated():
            st.success("Training is running — loss curve updating live")
        else:
            st.caption(f"Training data from {format_timestamp(LOSS_LOG)} (not currently running)")

        # Summary metrics
        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Step", f"{steps[-1]:,}")
        col2.metric("Latest Loss", f"{losses[-1]:.4f}")
        col3.metric("Starting Loss", f"{losses[0]:.2f}")
        col4.metric("Loss Reduction", f"{((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

        # Loss chart using Streamlit native charts
        import pandas as pd

        df = pd.DataFrame(loss_history)
        df = df.set_index("step")

        # Rename columns for display
        rename = {"loss": "Total Loss"}
        for col in df.columns:
            if col != "loss":
                rename[col] = col.replace("_", " ").title()
        df = df.rename(columns=rename)

        st.line_chart(df, use_container_width=True)

        # Show static plot if it exists
        plot_path = PLOTS_DIR / "training_loss.png"
        if plot_path.exists():
            with st.expander("Static training loss plot"):
                st.image(str(plot_path), use_container_width=True)


# ===================================================================
# PAGE: Evaluation Results
# ===================================================================
elif page == "Evaluation Results":
    st.header("Evaluation Results")

    eval_dirs = find_eval_dirs()

    if not eval_dirs:
        st.info(
            "No evaluation results yet. Run evaluation with:\n\n"
            "```bash\npython scripts/evaluate.py --checkpoint outputs/train/act_transfer_cube/last --device mps\n```"
        )
    else:
        # Pick which eval to view
        dir_names = [d.name for d in eval_dirs]
        selected = st.selectbox("Select evaluation run", dir_names)
        eval_path = EVAL_DIR / selected

        metrics = load_json(eval_path / "eval_metrics.json")
        if metrics is None:
            st.warning("No eval_metrics.json found in this directory.")
        else:
            summary = metrics["summary"]
            episodes = metrics["episodes"]

            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Success Rate", f"{summary['success_rate'] * 100:.0f}%")
            col2.metric("Avg Reward", f"{summary['avg_reward']:.2f}")
            col3.metric("Episodes", summary["n_episodes"])
            col4.metric("Checkpoint", summary.get("checkpoint", "unknown").split("/")[-1])

            # Episode table
            import pandas as pd
            ep_df = pd.DataFrame(episodes)
            ep_df.index = [f"Episode {i+1}" for i in range(len(ep_df))]
            if "elapsed_s" in ep_df.columns:
                ep_df["elapsed_s"] = ep_df["elapsed_s"].round(1)
            if "total_reward" in ep_df.columns:
                ep_df["total_reward"] = ep_df["total_reward"].round(2)
            st.dataframe(ep_df, use_container_width=True)

            # Success bar chart
            import pandas as pd
            success_data = pd.DataFrame({
                "Episode": [f"Ep {i+1}" for i in range(len(episodes))],
                "Reward": [ep["total_reward"] for ep in episodes],
            })
            st.bar_chart(success_data.set_index("Episode"), use_container_width=True)

        # Videos
        videos = find_videos(eval_path)
        if videos:
            st.subheader("Evaluation Videos")
            cols = st.columns(min(len(videos), 3))
            for i, video in enumerate(videos):
                with cols[i % 3]:
                    st.caption(video.name)
                    st.video(str(video))
        else:
            st.caption("No video files found in this evaluation run.")


# ===================================================================
# PAGE: Experiment History
# ===================================================================
elif page == "Experiment History":
    st.header("Experiment History")

    # Training runs
    st.subheader("Training Runs")
    if TRAIN_DIR.exists():
        # Find checkpoint directories
        checkpoints = sorted(
            [d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: d.name,
        )
        if checkpoints:
            import pandas as pd
            rows = []
            for ckpt in checkpoints:
                state_file = ckpt / "training_state.json"
                state = load_json(state_file)
                rows.append({
                    "Checkpoint": ckpt.name,
                    "Step": state.get("step", "?") if state else "?",
                    "Saved": format_timestamp(ckpt),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("No checkpoints saved yet.")

        # Loss log info
        if LOSS_LOG.exists():
            loss_data = load_json(LOSS_LOG)
            if loss_data:
                st.caption(
                    f"Loss log: {len(loss_data)} entries, "
                    f"steps {loss_data[0]['step']}-{loss_data[-1]['step']}, "
                    f"last updated {format_timestamp(LOSS_LOG)}"
                )
    else:
        st.info(
            "No training runs found. Start one with:\n\n"
            "```bash\npython scripts/train.py --steps 5000 --device mps\n```"
        )

    # Evaluation runs
    st.subheader("Evaluation Runs")
    eval_dirs = find_eval_dirs()
    if eval_dirs:
        import pandas as pd
        rows = []
        for d in eval_dirs:
            metrics = load_json(d / "eval_metrics.json")
            if metrics:
                s = metrics["summary"]
                rows.append({
                    "Name": d.name,
                    "Success Rate": f"{s['success_rate'] * 100:.0f}%",
                    "Avg Reward": f"{s['avg_reward']:.2f}",
                    "Episodes": s["n_episodes"],
                    "Date": format_timestamp(d),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No evaluation runs found.")


# ===================================================================
# PAGE: Baseline Comparison
# ===================================================================
elif page == "Baseline Comparison":
    st.header("Baseline Comparison")

    eval_dirs = find_eval_dirs()

    if len(eval_dirs) < 2:
        st.info(
            "Need at least 2 evaluation runs to compare. Run both trained and baseline evaluations:\n\n"
            "```bash\n"
            "# Evaluate trained model\n"
            "python scripts/evaluate.py --checkpoint outputs/train/act_transfer_cube/last --device mps\n\n"
            "# Evaluate pretrained baseline\n"
            "python scripts/evaluate.py --checkpoint lerobot/act_aloha_sim_transfer_cube_human --device mps\n"
            "```"
        )
    else:
        dir_names = [d.name for d in eval_dirs]

        col1, col2 = st.columns(2)
        with col1:
            model_a = st.selectbox("Model A", dir_names, index=0)
        with col2:
            default_b = 1 if len(dir_names) > 1 else 0
            model_b = st.selectbox("Model B", dir_names, index=default_b)

        metrics_a = load_json(EVAL_DIR / model_a / "eval_metrics.json")
        metrics_b = load_json(EVAL_DIR / model_b / "eval_metrics.json")

        if metrics_a and metrics_b:
            sa, sb = metrics_a["summary"], metrics_b["summary"]

            # Side by side metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(model_a)
                st.metric("Success Rate", f"{sa['success_rate'] * 100:.0f}%")
                st.metric("Avg Reward", f"{sa['avg_reward']:.2f}")
                st.metric("Episodes", sa["n_episodes"])
            with col2:
                st.subheader(model_b)
                st.metric("Success Rate", f"{sb['success_rate'] * 100:.0f}%")
                st.metric("Avg Reward", f"{sb['avg_reward']:.2f}")
                st.metric("Episodes", sb["n_episodes"])

            # Comparison bar chart
            import pandas as pd
            comp_df = pd.DataFrame({
                "Metric": ["Success Rate (%)", "Avg Reward"],
                model_a: [sa["success_rate"] * 100, sa["avg_reward"]],
                model_b: [sb["success_rate"] * 100, sb["avg_reward"]],
            }).set_index("Metric")
            st.bar_chart(comp_df, use_container_width=True)

            # Static comparison plot
            comp_plot = PLOTS_DIR / "comparison.png"
            if comp_plot.exists():
                with st.expander("Static comparison plot"):
                    st.image(str(comp_plot), use_container_width=True)
        else:
            st.warning("Could not load metrics for one or both selected models.")

    # Even with < 2 eval dirs, show the static comparison plot if it exists
    if len(eval_dirs) < 2:
        comp_plot = PLOTS_DIR / "comparison.png"
        if comp_plot.exists():
            st.subheader("Previous Comparison")
            st.image(str(comp_plot), use_container_width=True)


# ===================================================================
# PAGE: Status
# ===================================================================
elif page == "Status":
    st.header("System Status")

    # Training status
    st.subheader("Training")
    training_active = is_training_running()
    log_fresh = log_recently_updated()

    if training_active:
        st.success("Training process is running")
    elif log_fresh:
        st.warning("No training process detected, but log was recently updated")
    else:
        st.info("No training currently running")

    # Loss log status
    if LOSS_LOG.exists():
        loss_data = load_json(LOSS_LOG)
        if loss_data:
            st.caption(
                f"Loss log: {len(loss_data)} entries | "
                f"Steps {loss_data[0]['step']}-{loss_data[-1]['step']} | "
                f"Last updated: {format_timestamp(LOSS_LOG)}"
            )
    else:
        st.caption("No loss log found.")

    # Checkpoints
    st.subheader("Checkpoints")
    if TRAIN_DIR.exists():
        checkpoints = sorted(
            [d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("step_")],
        )
        if checkpoints:
            st.caption(f"{len(checkpoints)} checkpoint(s) saved")
            for ckpt in checkpoints:
                st.caption(f"  - {ckpt.name} ({format_timestamp(ckpt)})")
        else:
            st.caption("No checkpoints saved.")

        # Check for 'last' symlink
        last = TRAIN_DIR / "last"
        if last.exists():
            target = last.resolve().name if last.is_symlink() else "directory"
            st.caption(f"  'last' points to: {target}")
    else:
        st.caption("Training directory does not exist yet.")

    # Eval runs
    st.subheader("Evaluation Runs")
    eval_dirs = find_eval_dirs()
    if eval_dirs:
        st.caption(f"{len(eval_dirs)} evaluation run(s)")
        for d in eval_dirs:
            video_count = len(find_videos(d))
            st.caption(f"  - {d.name} ({format_timestamp(d)}, {video_count} videos)")
    else:
        st.caption("No evaluation runs found.")

    # Disk usage
    st.subheader("Output Directories")
    for name, path in [("Train", TRAIN_DIR), ("Eval", EVAL_DIR), ("Plots", PLOTS_DIR)]:
        if path.exists():
            # Count files
            files = list(path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            st.caption(f"{name}: {path} ({file_count} files)")
        else:
            st.caption(f"{name}: {path} (not created)")
