"""Interactive VLA Simulation Viewer with live probe execution.

Usage:
    .venv/bin/mjpython scripts/run_sim_interactive.py --scene widowx --model xvla
    .venv/bin/mjpython scripts/run_sim_interactive.py --scene franka --model pi0

Controls:
    Space / 1   — Baseline (~1s)
    2           — Spatial Symmetry: default + swapped (~2s)
    3           — Perturbation: 6 block positions (~8s)
    4           — Camera Sensitivity: normal + flipped + mirrored (~3s)
    5           — View Ablation: zero out each camera view (~4s)
    6           — Counterfactual: synonym prompt variants (~5s)
    7           — Null Action: "don't move" compliance (~9s)
    Arrow keys  — move red block
    C           — save camera frames to /tmp/ and open them
    R           — reset scene
    Q           — quit
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

KEY_SPACE = 32
KEY_R     = ord("R")
KEY_Q     = ord("Q")
KEY_UP    = 265
KEY_DOWN  = 264
KEY_LEFT  = 263
KEY_RIGHT = 262
KEY_1, KEY_2, KEY_3, KEY_4 = ord("1"), ord("2"), ord("3"), ord("4")
KEY_5, KEY_6, KEY_7 = ord("5"), ord("6"), ord("7")
KEY_C = ord("C")

BLOCK_STEP = 0.03

TRAJ_COLORS = [
    [0.2, 0.6, 1.0, 0.9],
    [1.0, 0.45, 0.1, 0.9],
    [0.2, 0.9, 0.3, 0.9],
    [0.9, 0.2, 0.9, 0.9],
    [0.9, 0.85, 0.1, 0.9],
    [0.5, 0.2, 1.0, 0.9],
]


def infer(adapter, scene, prompt, seed=0, flip_img=False, mirror_cam=False):
    """Single inference call — ~1s."""
    from vla_probing.adapter import VLAInput
    if mirror_cam:
        scene.mirror_camera()
    views = scene.render_all_views()
    if mirror_cam:
        scene.reset_camera()
    img = views["image"]
    if flip_img:
        img = np.fliplr(img).copy()
    proprio = (scene.get_joint_state()
               if (hasattr(adapter, "use_joint_state") and adapter.use_joint_state
                   and hasattr(scene, "get_joint_state"))
               else scene.get_ee_state())
    inp = VLAInput(images=[img, views["image2"]], prompt=prompt, proprio=proprio)
    adapter.reset()
    adapter.seed_for_inference(seed)
    t0 = time.time()
    output = adapter.predict_action(inp)
    elapsed = time.time() - t0
    actions = np.atleast_2d(output.actions)
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    return actions[:, :3], elapsed


def draw_trajectories(viewer, traj_list):
    """Draw list of (xyz, rgba) trajectories as spheres."""
    viewer.user_scn.ngeom = 0
    for xyz, color in traj_list:
        if xyz is None or len(xyz) == 0:
            continue
        n = min(len(xyz), viewer.user_scn.maxgeom - viewer.user_scn.ngeom - 2)
        for i in range(n):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
                break
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.012, 0.012]),
                xyz[i], np.eye(3).flatten(),
                np.array(color, dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom and len(xyz) > 0:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.022, 0.022, 0.022]),
                xyz[-1], np.eye(3).flatten(),
                np.array(color[:3] + [1.0], dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1


def add_ee_marker(viewer, pos):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.025, 0.025, 0.025]),
        pos, np.eye(3).flatten(),
        np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  default="widowx", choices=["widowx", "franka"])
    parser.add_argument("--model",  default="xvla",   choices=["xvla", "pi0", "openvla_oft", "cosmos_policy"])
    parser.add_argument("--prompt", default="pick up the red block")
    parser.add_argument("--device", default="mps",    choices=["mps", "cpu", "cuda"])
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}...")
    from vla_probing.scene import make_scene
    scene = make_scene(args.scene)
    scene.reset()

    print(f"Loading model: {args.model}...")
    from vla_probing.probes.base import make_adapter
    adapter = make_adapter(args.model, args.device)

    print(f"\nReady. Prompt: \"{args.prompt}\"")
    print("  1/Spc  Baseline (~1s)")
    print("  2      Spatial Symmetry (~2s)")
    print("  3      Perturbation — 6 positions (~8s)")
    print("  4      Camera Sensitivity (~3s)")
    print("  5      View Ablation — zero out views (~4s)")
    print("  6      Counterfactual — synonym prompts (~5s)")
    print("  7      Null Action — 'don't move' compliance (~9s)")
    print("  C      Save camera frames  |  Arrows  Move red block  |  R Reset  |  Q Quit\n")

    state = {
        "probe":    None,
        "reset":    False,
        "quit":     False,
        "move":     None,
        "running":  False,
        "trajs":    [],
        "save_cam": False,
    }

    def key_callback(keycode):
        if state["running"]:
            return
        if   keycode in (KEY_SPACE, KEY_1): state["probe"] = "baseline"
        elif keycode == KEY_2:              state["probe"] = "spatial"
        elif keycode == KEY_3:              state["probe"] = "perturbation"
        elif keycode == KEY_4:              state["probe"] = "camera"
        elif keycode == KEY_5:              state["probe"] = "view_ablation"
        elif keycode == KEY_6:              state["probe"] = "counterfactual"
        elif keycode == KEY_7:              state["probe"] = "null_action"
        elif keycode == KEY_C:              state["save_cam"] = True
        elif keycode == KEY_R:              state["reset"] = True
        elif keycode == KEY_Q:              state["quit"]  = True
        elif keycode == KEY_UP:             state["move"]  = (0, +BLOCK_STEP)
        elif keycode == KEY_DOWN:           state["move"]  = (0, -BLOCK_STEP)
        elif keycode == KEY_RIGHT:          state["move"]  = (1, -BLOCK_STEP)
        elif keycode == KEY_LEFT:           state["move"]  = (1, +BLOCK_STEP)

    def run_probe_thread(probe_name):
        from vla_probing.tracking import ExperimentTracker
        from vla_probing.probes.baseline import BaselineProbe
        from vla_probing.probes.spatial_symmetry import SpatialSymmetryProbe
        from vla_probing.probes.perturbation import PerturbationProbe
        from vla_probing.probes.camera_sensitivity import CameraSensitivityProbe
        from vla_probing.probes.view_ablation import ViewAblationProbe
        from vla_probing.probes.counterfactual import CounterfactualProbe
        from vla_probing.probes.null_action import NullActionProbe
        tracker = ExperimentTracker(enabled=False)

        PROBE_MAP = {
            "baseline":      (BaselineProbe,          "[1] Baseline"),
            "spatial":       (SpatialSymmetryProbe,   "[2] Spatial Symmetry"),
            "perturbation":  (PerturbationProbe,      "[3] Perturbation"),
            "camera":        (CameraSensitivityProbe, "[4] Camera Sensitivity"),
            "view_ablation": (ViewAblationProbe,      "[5] View Ablation"),
            "counterfactual":(CounterfactualProbe,    "[6] Counterfactual"),
            "null_action":   (NullActionProbe,        "[7] Null Action"),
        }
        # Trajectory artifact keys to visualize (probes without XYZ just print metrics)
        TRAJ_KEYS = {
            "baseline":      [("trajectory_xyz", 0)],
            "spatial":       [("baseline_xyz", 0), ("swapped_xyz", 1)],
            "perturbation":  [("baseline_xyz", 0), ("traj_xyz_shifted_left", 1),
                              ("traj_xyz_shifted_right", 2), ("traj_xyz_shifted_forward", 3),
                              ("traj_xyz_shifted_back", 4), ("traj_xyz_raised", 5)],
            "camera":        [("baseline_xyz", 0), ("flipped_xyz", 1), ("mirrored_xyz", 2)],
            "view_ablation": [],  # no XYZ artifacts — metrics only
            "counterfactual":[],  # no XYZ artifacts — metrics only
            "null_action":   [],  # no XYZ artifacts — metrics only
        }

        state["trajs"] = []
        probe_cls, label = PROBE_MAP[probe_name]
        print(f"\n{label} running (n_seeds=1)...")
        t0 = time.time()

        # Give the probe its own MjData so viewer's scene.data is never
        # written during probe execution (concurrent write+read corrupts state
        # and causes the arm-disappear / crash / wrong-coords bugs).
        probe_data = mujoco.MjData(scene.model)
        probe_data.qpos[:] = scene.data.qpos  # carry current block positions
        mujoco.mj_forward(scene.model, probe_data)
        original_data = scene.data
        scene.data = probe_data

        try:
            probe = probe_cls(adapter, scene, tracker)
            result = probe.run(seed=0, n_seeds=1)
            elapsed = time.time() - t0

            # Print all metrics
            print(f"  done in {elapsed:.1f}s")
            for k, v in result.metrics.items():
                print(f"  {k}: {v:.4f}")

            # Extract trajectories for visualisation
            trajs = []
            for key, color_idx in TRAJ_KEYS[probe_name]:
                xyz = result.artifacts.get(key)
                if xyz is not None and len(xyz) > 0:
                    trajs.append((xyz, TRAJ_COLORS[color_idx]))
            state["trajs"] = trajs

            if probe_name == "spatial":
                print("  blue=baseline  orange=swapped")
            elif probe_name == "perturbation":
                print("  blue=default  orange=left  green=right  pink=fwd  yellow=back  purple=raised")
            elif probe_name == "camera":
                print("  blue=baseline  orange=flipped  green=mirrored")
            elif probe_name in ("view_ablation", "counterfactual", "null_action"):
                print("  (metrics only — no trajectory visualization for this probe)")

        except Exception as e:
            import traceback; traceback.print_exc()
        finally:
            scene.data = original_data  # always restore viewer's data
            state["running"] = False
            state["probe"] = None

    with mujoco.viewer.launch_passive(
        scene.model, scene.data, key_callback=key_callback
    ) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.1]
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        while viewer.is_running():
            if state["quit"]:
                break

            if state["move"] and not state["running"]:
                axis, delta = state["move"]
                pos = scene.get_block_pos("red").copy()
                pos[axis] += delta
                scene.set_red_block_pos(pos)
                state["move"] = None
                state["trajs"] = []
                print(f"Red block → X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f}")

            if state["reset"] and not state["running"]:
                scene.reset()
                state["trajs"] = []
                state["reset"] = False
                print("Reset.")

            if state["save_cam"]:
                from PIL import Image
                views = scene.render_all_views()
                p1 = "/tmp/vla_cam_primary.png"
                p2 = "/tmp/vla_cam_secondary.png"
                Image.fromarray(views["image"]).save(p1)
                Image.fromarray(views["image2"]).save(p2)
                import subprocess
                subprocess.Popen(["open", p1, p2])
                print(f"Camera frames saved → {p1}  {p2}")
                state["save_cam"] = False

            if state["probe"] and not state["running"]:
                probe_to_run = state["probe"]
                state["probe"] = None
                state["running"] = True
                # Run synchronously on this thread — avoids creating a second CGL
                # context (which would happen in a background thread via the TLS
                # renderer path) and eliminates the macOS CGL interference that
                # causes the arm to disappear from the viewer.
                run_probe_thread(probe_to_run)
                # state["running"] = False set inside run_probe_thread's finally block

            viewer.user_scn.ngeom = 0
            if state["trajs"]:
                draw_trajectories(viewer, state["trajs"])
            if not state["running"]:
                add_ee_marker(viewer, scene.get_ee_state()[:3])
                mujoco.mj_step(scene.model, scene.data)
                viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
