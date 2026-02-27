"""MuJoCo scene renderers for VLA probing.

Provides WidowXScene and FrankaScene — each loads the respective MuJoCo
scene, renders camera views, and exposes the same interface for
block manipulation, camera mirroring, and end-effector state readout.
"""

from pathlib import Path

import mujoco
import numpy as np

# ─── Shared constants ────────────────────────────────────────────────
RENDER_WIDTH = 256
RENDER_HEIGHT = 256

# ─── WidowX constants ───────────────────────────────────────────────
WIDOWX_ASSETS_DIR = Path(__file__).parent / "assets" / "widowx"
WIDOWX_SCENE_XML = WIDOWX_ASSETS_DIR / "widowx_vision_scene.xml"
WIDOWX_CAMERA_PRIMARY = "up"
WIDOWX_CAMERA_SECONDARY = "side"
WIDOWX_HOME_QPOS = np.array([0.0, -0.56, 0.76, 0.0, 1.27, 0.0, 0.015, -0.015])

# ─── Franka constants ───────────────────────────────────────────────
FRANKA_ASSETS_DIR = Path(__file__).parent / "assets" / "franka"
FRANKA_SCENE_XML = FRANKA_ASSETS_DIR / "franka_vision_scene.xml"
FRANKA_CAMERA_PRIMARY = "agentview"
FRANKA_CAMERA_SECONDARY = "robot0_eye_in_hand"
# Franka Panda home qpos: 7 arm joints + 2 finger joints (from menagerie keyframe)
FRANKA_HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04]
)

# ─── Backward-compat aliases used by existing imports ────────────────
ASSETS_DIR = WIDOWX_ASSETS_DIR
SCENE_XML = WIDOWX_SCENE_XML
CAMERA_PRIMARY = WIDOWX_CAMERA_PRIMARY
CAMERA_SECONDARY = WIDOWX_CAMERA_SECONDARY
CAMERA_DEBUG = "third_person"
HOME_QPOS = WIDOWX_HOME_QPOS
RED_BLOCK_DEFAULT_POS = np.array([0.25, 0.1, 0.02])
BLUE_BLOCK_DEFAULT_POS = np.array([0.25, -0.1, 0.02])


# =====================================================================
# WidowXScene
# =====================================================================
class WidowXScene:
    """MuJoCo WidowX scene for VLA probing experiments."""

    scene_name = "widowx"
    primary_camera = WIDOWX_CAMERA_PRIMARY
    secondary_camera = WIDOWX_CAMERA_SECONDARY

    def __init__(
        self,
        width: int = RENDER_WIDTH,
        height: int = RENDER_HEIGHT,
        scene_xml: Path | None = None,
    ) -> None:
        xml_path = scene_xml or WIDOWX_SCENE_XML
        self._scene_xml = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height

        # Body/joint IDs for manipulation
        self._red_block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
        )
        self._blue_block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_block"
        )

        self.reset()

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset scene to home configuration."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: len(qpos)] = qpos
        else:
            self.data.qpos[:8] = WIDOWX_HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str = WIDOWX_CAMERA_PRIMARY) -> np.ndarray:
        """Render the scene from the specified camera."""
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_all_views(self) -> dict[str, np.ndarray]:
        """Render from both primary and secondary cameras."""
        return {
            "image": self.render(self.primary_camera),
            "image2": self.render(self.secondary_camera),
        }

    def get_ee_state(self) -> np.ndarray:
        """Get end-effector state in BridgeData format.

        Returns 8D vector: [x, y, z, roll, pitch, yaw, 0, gripper_pos]
        """
        ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )

        if ee_site_id >= 0:
            ee_pos = self.data.site_xpos[ee_site_id].copy()
            ee_mat = self.data.site_xmat[ee_site_id].reshape(3, 3)
        else:
            ee_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_bar_link"
            )
            ee_pos = self.data.xpos[ee_body].copy()
            ee_mat = self.data.xmat[ee_body].reshape(3, 3)

        ee_euler = _rotmat_to_euler(ee_mat)
        gripper_pos = self.data.qpos[6]
        return np.array(
            [
                ee_pos[0], ee_pos[1], ee_pos[2],
                ee_euler[0], ee_euler[1], ee_euler[2],
                0.0, gripper_pos,
            ]
        )

    def set_red_block_pos(self, pos: np.ndarray) -> None:
        """Move the red block to a new position."""
        if self._red_block_body < 0:
            return
        joint_id = self.model.body_jntadr[self._red_block_body]
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            mujoco.mj_forward(self.model, self.data)

    def set_blue_block_pos(self, pos: np.ndarray) -> None:
        """Move the blue block to a new position (if it has a freejoint)."""
        if self._blue_block_body < 0:
            return
        joint_id = self.model.body_jntadr[self._blue_block_body]
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            mujoco.mj_forward(self.model, self.data)

    def swap_block_positions(self) -> None:
        """Swap red and blue block positions for spatial symmetry probe."""
        red_pos = self.get_block_pos("red")
        blue_pos = self.get_block_pos("blue")
        if red_pos is not None and blue_pos is not None:
            self.set_red_block_pos(blue_pos.copy())
            self.set_blue_block_pos(red_pos.copy())

    def get_block_pos(self, color: str) -> np.ndarray | None:
        """Get current position of a block."""
        body_id = (
            self._red_block_body if color == "red" else self._blue_block_body
        )
        if body_id < 0:
            return None
        return self.data.xpos[body_id].copy()

    def mirror_camera(self, camera_name: str | None = None) -> None:
        """Mirror the camera view horizontally (negate Y position)."""
        camera_name = camera_name or self.primary_camera
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id, 1] *= -1

    def reset_camera(self, camera_name: str | None = None) -> None:
        """Reset camera to original position (reload model params)."""
        camera_name = camera_name or self.primary_camera
        original = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id] = original.cam_pos[cam_id]
            self.model.cam_quat[cam_id] = original.cam_quat[cam_id]

    def render_with_trajectory(
        self,
        trajectory_xyz: np.ndarray,
        camera: str | None = None,
    ) -> np.ndarray:
        """Render scene with trajectory markers overlaid."""
        camera = camera or self.primary_camera
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        cam.fixedcamid = cam_id

        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene,
        )

        n_points = len(trajectory_xyz)
        for i, pos in enumerate(trajectory_xyz):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=pos.astype(np.float64),
                mat=np.eye(3).flatten().astype(np.float64),
                rgba=_trajectory_color(i, n_points),
            )
            scene.ngeom += 1

        self.renderer.update_scene(self.data, camera=camera)
        ctx = mujoco.gl_context.GLContext(self.width, self.height)
        ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, scene, ctx)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, viewport, ctx)
        img = np.flipud(img)
        ctx.free()
        return img

    def close(self) -> None:
        """Clean up renderer resources."""
        self.renderer.close()


# =====================================================================
# FrankaScene
# =====================================================================
class FrankaScene:
    """MuJoCo Franka Panda scene for VLA probing (LIBERO-style)."""

    scene_name = "franka"
    primary_camera = FRANKA_CAMERA_PRIMARY
    secondary_camera = FRANKA_CAMERA_SECONDARY

    def __init__(
        self,
        width: int = RENDER_WIDTH,
        height: int = RENDER_HEIGHT,
        scene_xml: Path | None = None,
    ) -> None:
        xml_path = scene_xml or FRANKA_SCENE_XML
        self._scene_xml = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height

        # Body IDs
        self._hand_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        self._red_block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
        )
        self._blue_block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_block"
        )

        # Wrist camera ID (updated to track hand body before rendering)
        self._wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, FRANKA_CAMERA_SECONDARY
        )

        self.reset()

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset scene to home configuration."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: len(qpos)] = qpos
        else:
            self.data.qpos[:9] = FRANKA_HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self._sync_wrist_camera()

    def _sync_wrist_camera(self) -> None:
        """Move the wrist camera to track the hand body's pose."""
        if self._wrist_cam_id < 0 or self._hand_body < 0:
            return
        hand_pos = self.data.xpos[self._hand_body].copy()
        hand_quat = self.data.xquat[self._hand_body].copy()
        # Offset camera slightly below the hand (looking down at table)
        self.model.cam_pos[self._wrist_cam_id] = hand_pos + np.array([0.0, 0.0, -0.05])

    def render(self, camera: str = FRANKA_CAMERA_PRIMARY) -> np.ndarray:
        """Render the scene from the specified camera."""
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_all_views(self) -> dict[str, np.ndarray]:
        """Render from both primary and secondary cameras."""
        return {
            "image": self.render(self.primary_camera),
            "image2": self.render(self.secondary_camera),
        }

    def get_ee_state(self) -> np.ndarray:
        """Get end-effector state in LIBERO format.

        Returns 8D vector: [x, y, z, qx, qy, qz, qw, gripper_pos]

        LIBERO uses quaternion orientation (not Euler) for the Franka EE.
        The Pi0 adapter maps this to its internal state representation.
        """
        ee_pos = self.data.xpos[self._hand_body].copy()
        ee_quat = self.data.xquat[self._hand_body].copy()  # (w, x, y, z)

        # MuJoCo quat is (w, x, y, z); LIBERO expects (x, y, z, w)
        qx, qy, qz, qw = ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]

        gripper_pos = self.data.qpos[7]  # finger_joint1 position
        return np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            qx, qy, qz, qw,
            gripper_pos,
        ])

    def set_red_block_pos(self, pos: np.ndarray) -> None:
        """Move the red block to a new position."""
        if self._red_block_body < 0:
            return
        joint_id = self.model.body_jntadr[self._red_block_body]
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            mujoco.mj_forward(self.model, self.data)

    def set_blue_block_pos(self, pos: np.ndarray) -> None:
        """Move the blue block to a new position (if it has a freejoint)."""
        if self._blue_block_body < 0:
            return
        joint_id = self.model.body_jntadr[self._blue_block_body]
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            mujoco.mj_forward(self.model, self.data)

    def swap_block_positions(self) -> None:
        """Swap red and blue block positions."""
        red_pos = self.get_block_pos("red")
        blue_pos = self.get_block_pos("blue")
        if red_pos is not None and blue_pos is not None:
            self.set_red_block_pos(blue_pos.copy())
            self.set_blue_block_pos(red_pos.copy())

    def get_block_pos(self, color: str) -> np.ndarray | None:
        """Get current position of a block."""
        body_id = (
            self._red_block_body if color == "red" else self._blue_block_body
        )
        if body_id < 0:
            return None
        return self.data.xpos[body_id].copy()

    def mirror_camera(self, camera_name: str | None = None) -> None:
        """Mirror the camera view horizontally (negate Y position)."""
        camera_name = camera_name or self.primary_camera
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id, 1] *= -1

    def reset_camera(self, camera_name: str | None = None) -> None:
        """Reset camera to original position (reload model params)."""
        camera_name = camera_name or self.primary_camera
        original = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id] = original.cam_pos[cam_id]
            self.model.cam_quat[cam_id] = original.cam_quat[cam_id]

    def render_with_trajectory(
        self,
        trajectory_xyz: np.ndarray,
        camera: str | None = None,
    ) -> np.ndarray:
        """Render scene with trajectory markers overlaid."""
        camera = camera or self.primary_camera
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        cam.fixedcamid = cam_id

        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene,
        )

        n_points = len(trajectory_xyz)
        for i, pos in enumerate(trajectory_xyz):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=pos.astype(np.float64),
                mat=np.eye(3).flatten().astype(np.float64),
                rgba=_trajectory_color(i, n_points),
            )
            scene.ngeom += 1

        self.renderer.update_scene(self.data, camera=camera)
        ctx = mujoco.gl_context.GLContext(self.width, self.height)
        ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, scene, ctx)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, viewport, ctx)
        img = np.flipud(img)
        ctx.free()
        return img

    def close(self) -> None:
        """Clean up renderer resources."""
        self.renderer.close()


# =====================================================================
# Scene factory
# =====================================================================
# Union type for type hints
Scene = WidowXScene | FrankaScene

# Auto-detect best scene for each model
_MODEL_DEFAULT_SCENE = {
    "pi0": "franka",
    "xvla": "widowx",
    "openvla": "widowx",
    "cosmos_policy": "franka",
    "groot": "franka",
}


def make_scene(scene_name: str = "widowx") -> Scene:
    """Factory to create a scene by name."""
    scenes = {
        "widowx": WidowXScene,
        "franka": FrankaScene,
    }
    if scene_name not in scenes:
        raise ValueError(
            f"Unknown scene: {scene_name}. Available: {list(scenes.keys())}"
        )
    return scenes[scene_name]()


def default_scene_for_model(model: str) -> str:
    """Return the native scene name for a given model."""
    return _MODEL_DEFAULT_SCENE.get(model, "widowx")


# =====================================================================
# Helpers
# =====================================================================
def _trajectory_color(index: int, total: int) -> np.ndarray:
    """Generate green-to-red gradient color for trajectory point."""
    t = index / max(total - 1, 1)
    return np.array([t, 1.0 - t, 0.0, 0.8], dtype=np.float32)


def _rotmat_to_euler(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to XYZ Euler angles."""
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])
