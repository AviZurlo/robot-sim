"""Interactive VLA Simulation Viewer — browser-based, no display required.

Run on Mac Mini:
    cd ~/Projects/robot-sim
    .venv/bin/streamlit run scripts/sim_viewer.py --server.port 8502

Open http://100.87.81.72:8502 on your MacBook Pro.
"""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="VLA Sim Viewer", layout="wide")

# ── Block position defaults ───────────────────────────────────────────────
DEFAULTS = {
    "widowx": {"red": [0.25, 0.10, 0.02], "blue": [0.25, -0.10, 0.02]},
    "franka": {"red": [0.45, -0.03, 0.08], "blue": [0.45, 0.15, 0.08]},
}
RANGES = {
    "widowx": {"x": (0.10, 0.50), "y": (-0.28, 0.30), "z": (0.02, 0.15)},
    "franka": {"x": (0.25, 0.65), "y": (-0.28, 0.35), "z": (0.05, 0.20)},
}

# ── Session state init ───────────────────────────────────────────────────
def _init_state():
    defaults = {
        "scene_type":   "widowx",
        "red_pos":      DEFAULTS["widowx"]["red"][:],
        "blue_pos":     DEFAULTS["widowx"]["blue"][:],
        "mirror_cam":   False,
        "flip_img":     False,
        "running":      False,
        "step":         0,
        "prediction":   None,
        "exec_step":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


@st.cache_resource
def load_scene(scene_name: str):
    from vla_probing.scene import make_scene
    s = make_scene(scene_name)
    s.reset()
    return s


@st.cache_resource
def load_adapter(model_name: str):
    from vla_probing.probes.base import make_adapter
    return make_adapter(model_name, device="mps")


def render_current(scene, mirror_cam=False, flip_img=False):
    if mirror_cam:
        scene.mirror_camera()
    views = scene.render_all_views()
    if mirror_cam:
        scene.reset_camera()
    img = views["image"]
    if flip_img:
        img = np.fliplr(img).copy()
    return img, views["image2"]


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VLA Sim Viewer")

    # Scene picker
    new_scene = st.radio("Scene", ["widowx", "franka"], horizontal=True,
                         index=["widowx", "franka"].index(st.session_state.scene_type))
    if new_scene != st.session_state.scene_type:
        st.session_state.scene_type = new_scene
        st.session_state.red_pos  = DEFAULTS[new_scene]["red"][:]
        st.session_state.blue_pos = DEFAULTS[new_scene]["blue"][:]
        st.session_state.running  = False
        st.session_state.step     = 0
        st.rerun()

    scene_type = st.session_state.scene_type
    rng = RANGES[scene_type]

    st.divider()

    # Block position sliders — read current values from session_state as defaults
    st.markdown("**Red block**")
    rx = st.slider("rx", *rng["x"], value=float(st.session_state.red_pos[0]),  step=0.01, label_visibility="collapsed", key="_rx")
    ry = st.slider("ry", *rng["y"], value=float(st.session_state.red_pos[1]),  step=0.01, label_visibility="collapsed", key="_ry")
    rz = st.slider("rz", *rng["z"], value=float(st.session_state.red_pos[2]),  step=0.01, label_visibility="collapsed", key="_rz")
    st.caption(f"X {rx:.2f}  Y {ry:.2f}  Z {rz:.2f}")

    st.markdown("**Blue block**")
    bx = st.slider("bx", *rng["x"], value=float(st.session_state.blue_pos[0]), step=0.01, label_visibility="collapsed", key="_bx")
    by = st.slider("by", *rng["y"], value=float(st.session_state.blue_pos[1]), step=0.01, label_visibility="collapsed", key="_by")
    bz = st.slider("bz", *rng["z"], value=float(st.session_state.blue_pos[2]), step=0.01, label_visibility="collapsed", key="_bz")
    st.caption(f"X {bx:.2f}  Y {by:.2f}  Z {bz:.2f}")

    st.session_state.red_pos  = [rx, ry, rz]
    st.session_state.blue_pos = [bx, by, bz]

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Reset", use_container_width=True):
        d = DEFAULTS[scene_type]
        st.session_state.red_pos  = d["red"][:]
        st.session_state.blue_pos = d["blue"][:]
        st.session_state.step     = 0
        st.session_state.running  = False
        st.rerun()
    if c2.button("Swap ↔", use_container_width=True):
        st.session_state.red_pos, st.session_state.blue_pos = (
            st.session_state.blue_pos[:], st.session_state.red_pos[:])
        st.rerun()

    st.divider()
    mirror_cam = st.checkbox("Mirror camera", value=st.session_state.mirror_cam)
    flip_img   = st.checkbox("Flip image",    value=st.session_state.flip_img)
    st.session_state.mirror_cam = mirror_cam
    st.session_state.flip_img   = flip_img


# ── Apply positions to scene ─────────────────────────────────────────────
scene = load_scene(scene_type)
scene.reset()
scene.set_red_block_pos(np.array(st.session_state.red_pos))
scene.set_blue_block_pos(np.array(st.session_state.blue_pos))

# ── Main panel ───────────────────────────────────────────────────────────
st.markdown("### Camera views")
img_primary, img_secondary = render_current(scene, mirror_cam, flip_img)
cam1, cam2 = st.columns(2)
primary_placeholder   = cam1.empty()
secondary_placeholder = cam2.empty()
primary_placeholder.image(img_primary,   caption="Primary", use_container_width=True)
secondary_placeholder.image(img_secondary, caption="Secondary", use_container_width=True)

# ── Quick probe buttons ───────────────────────────────────────────────────
st.markdown("### Probe shortcuts")
p1, p2, p3, p4, p5, p6 = st.columns(6)

if p1.button("Baseline", use_container_width=True):
    d = DEFAULTS[scene_type]
    st.session_state.red_pos    = d["red"][:]
    st.session_state.blue_pos   = d["blue"][:]
    st.session_state.mirror_cam = False
    st.session_state.flip_img   = False
    st.rerun()

if p2.button("Swap blocks", use_container_width=True):
    st.session_state.red_pos, st.session_state.blue_pos = (
        st.session_state.blue_pos[:], st.session_state.red_pos[:])
    st.rerun()

if p3.button("Mirror cam", use_container_width=True):
    st.session_state.mirror_cam = not st.session_state.mirror_cam
    st.rerun()

if p4.button("Flip image", use_container_width=True):
    st.session_state.flip_img = not st.session_state.flip_img
    st.rerun()

if p5.button("Red left +8cm", use_container_width=True):
    pos = st.session_state.red_pos[:]
    pos[1] += 0.08
    st.session_state.red_pos = pos
    st.rerun()

if p6.button("Red forward +5cm", use_container_width=True):
    pos = st.session_state.red_pos[:]
    pos[0] += 0.05
    st.session_state.red_pos = pos
    st.rerun()

# ── Physics simulation ───────────────────────────────────────────────────
st.divider()
st.markdown("### Live physics simulation")
st.caption("Steps MuJoCo physics and streams frames to your browser in real time.")

phys_c1, phys_c2, phys_c3, phys_c4 = st.columns(4)
start_btn  = phys_c1.button("▶ Run",   type="primary", use_container_width=True)
stop_btn   = phys_c2.button("⏹ Stop",  use_container_width=True)
step1_btn  = phys_c3.button("Step ×1", use_container_width=True)
step10_btn = phys_c4.button("Step ×10",use_container_width=True)

fps_label   = st.empty()
step_label  = st.empty()

if stop_btn:
    st.session_state.running = False

if step1_btn:
    mujoco.mj_step(scene.model, scene.data)
    st.session_state.step += 1
    img, img2 = render_current(scene, mirror_cam, flip_img)
    primary_placeholder.image(img,   caption="Primary",   use_container_width=True)
    secondary_placeholder.image(img2, caption="Secondary", use_container_width=True)
    step_label.caption(f"Step {st.session_state.step}")

if step10_btn:
    for _ in range(10):
        mujoco.mj_step(scene.model, scene.data)
    st.session_state.step += 10
    img, img2 = render_current(scene, mirror_cam, flip_img)
    primary_placeholder.image(img,   caption="Primary",   use_container_width=True)
    secondary_placeholder.image(img2, caption="Secondary", use_container_width=True)
    step_label.caption(f"Step {st.session_state.step}")

if start_btn:
    st.session_state.running = True

if st.session_state.running:
    fps_target = 10
    n_frames   = 300  # run for ~30s then pause to prevent runaway
    t_start    = time.time()
    frames_rendered = 0

    while st.session_state.running and frames_rendered < n_frames:
        t0 = time.time()

        # Physics steps per render (sub-step for stability)
        for _ in range(5):
            mujoco.mj_step(scene.model, scene.data)
        st.session_state.step += 5

        img, img2 = render_current(scene, mirror_cam, flip_img)
        primary_placeholder.image(img,    caption="Primary",   use_container_width=True)
        secondary_placeholder.image(img2, caption="Secondary", use_container_width=True)

        frames_rendered += 1
        elapsed = time.time() - t0
        actual_fps = 1.0 / max(elapsed, 1e-6)
        fps_label.caption(f"~{actual_fps:.0f} FPS · step {st.session_state.step}")

        sleep_time = max(0.0, 1.0 / fps_target - elapsed)
        time.sleep(sleep_time)

    if frames_rendered >= n_frames:
        st.session_state.running = False
        st.rerun()

# ── VLA Inference ────────────────────────────────────────────────────────
st.divider()
st.markdown("### Run a VLA prediction")

inf_c1, inf_c2 = st.columns([3, 1])
prompt     = inf_c1.text_input("Prompt", "pick up the red block")
model_name = inf_c2.selectbox("Model", ["xvla", "pi0"],
                               index=0 if scene_type == "widowx" else 1)

run_inf = st.button("▶ Run inference + execute in sim", type="primary")

if run_inf:
    with st.spinner(f"Running {model_name}... (loads model first time, ~1 min)"):
        try:
            adapter = load_adapter(model_name)
            from vla_probing.adapter import VLAInput

            views = scene.render_all_views()
            proprio = (scene.get_joint_state()
                      if (hasattr(adapter, "use_joint_state") and adapter.use_joint_state
                          and hasattr(scene, "get_joint_state"))
                      else scene.get_ee_state())

            inp = VLAInput(images=[views["image"], views["image2"]],
                          prompt=prompt, proprio=proprio)
            adapter.reset()
            adapter.seed_for_inference(0)
            output = adapter.predict_action(inp)

            actions = np.atleast_2d(output.actions)
            if actions.ndim == 3:
                actions = actions.reshape(-1, actions.shape[-1])

            st.session_state.prediction = actions
            st.session_state.exec_step  = 0
            st.success(f"Got {len(actions)} action steps (dim={actions.shape[1]})")
        except Exception as e:
            st.error(str(e))

# Execute predicted actions in sim, streaming frames
if st.session_state.prediction is not None:
    actions = st.session_state.prediction
    exec_step = st.session_state.exec_step
    remaining = len(actions) - exec_step

    st.info(f"Predicted {len(actions)} steps — {remaining} remaining. "
            f"Press **Execute** to play through them.")

    exec_btn = st.button("▶ Execute actions in sim")
    if exec_btn:
        st.markdown("**Executing predicted trajectory...**")
        prog = st.progress(0)

        for i in range(exec_step, len(actions)):
            action = actions[i]
            # Apply action as position targets / deltas
            if len(scene.data.ctrl) > 0:
                n = min(len(action), len(scene.data.ctrl))
                scene.data.ctrl[:n] = action[:n]
            mujoco.mj_step(scene.model, scene.data)

            img, img2 = render_current(scene, mirror_cam, flip_img)
            primary_placeholder.image(img,    caption="Primary",   use_container_width=True)
            secondary_placeholder.image(img2, caption="Secondary", use_container_width=True)

            prog.progress((i + 1) / len(actions))
            time.sleep(0.05)

        st.session_state.exec_step  = len(actions)
        st.session_state.prediction = None
        st.success("Done.")
