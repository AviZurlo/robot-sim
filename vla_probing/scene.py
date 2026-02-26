"""MuJoCo WidowX scene renderer for VLA probing.

Handles loading the WidowX MuJoCo scene, rendering camera views,
and modifying scene parameters for diagnostic probes.
"""

from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets" / "widowx"
SCENE_XML = ASSETS_DIR / "widowx_vision_scene.xml"

# Camera names matching BridgeData conventions
CAMERA_PRIMARY = "up"  # over-the-right-shoulder view
CAMERA_SECONDARY = "side"  # angled side view
CAMERA_DEBUG = "third_person"  # debug/overview camera

# Default render resolution for VLA input
RENDER_WIDTH = 256
RENDER_HEIGHT = 256

# WidowX home qpos: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, left_finger, right_finger]
HOME_QPOS = np.array([0.0, -0.56, 0.76, 0.0, 1.27, 0.0, 0.015, -0.015])

# Default block positions (from the scene XML)
RED_BLOCK_DEFAULT_POS = np.array([0.25, 0.1, 0.02])
BLUE_BLOCK_DEFAULT_POS = np.array([0.25, -0.1, 0.02])


class WidowXScene:
    """MuJoCo WidowX scene for VLA probing experiments."""

    def __init__(
        self,
        width: int = RENDER_WIDTH,
        height: int = RENDER_HEIGHT,
        scene_xml: Path | None = None,
    ) -> None:
        xml_path = scene_xml or SCENE_XML
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
            self.data.qpos[:8] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str = CAMERA_PRIMARY) -> np.ndarray:
        """Render the scene from the specified camera.

        Returns:
            RGB image as (H, W, 3) uint8 numpy array.
        """
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_all_views(self) -> dict[str, np.ndarray]:
        """Render from both primary and secondary cameras."""
        return {
            "image": self.render(CAMERA_PRIMARY),
            "image2": self.render(CAMERA_SECONDARY),
        }

    def get_ee_state(self) -> np.ndarray:
        """Get end-effector state in BridgeData format.

        Returns 8D vector: [x, y, z, roll, pitch, yaw, 0, gripper_pos]
        """
        # Get EE site position (gripper center)
        ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )

        if ee_site_id >= 0:
            ee_pos = self.data.site_xpos[ee_site_id].copy()
            ee_mat = self.data.site_xmat[ee_site_id].reshape(3, 3)
        else:
            # Fallback: use last link body
            ee_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_bar_link"
            )
            ee_pos = self.data.xpos[ee_body].copy()
            ee_mat = self.data.xmat[ee_body].reshape(3, 3)

        # Convert rotation matrix to euler angles
        ee_euler = _rotmat_to_euler(ee_mat)

        gripper_pos = self.data.qpos[6]  # left finger joint position
        return np.array(
            [
                ee_pos[0],
                ee_pos[1],
                ee_pos[2],
                ee_euler[0],
                ee_euler[1],
                ee_euler[2],
                0.0,
                gripper_pos,
            ]
        )

    def set_red_block_pos(self, pos: np.ndarray) -> None:
        """Move the red block to a new position."""
        if self._red_block_body < 0:
            return
        # Red block has a freejoint, so its qpos starts after the robot joints
        # Find the freejoint qpos index
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

    def mirror_camera(self, camera_name: str = CAMERA_PRIMARY) -> None:
        """Mirror the camera view horizontally (negate Y position)."""
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id, 1] *= -1

    def reset_camera(self, camera_name: str = CAMERA_PRIMARY) -> None:
        """Reset camera to original position (reload model params)."""
        # Re-read from the original model spec
        original = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            self.model.cam_pos[cam_id] = original.cam_pos[cam_id]
            self.model.cam_quat[cam_id] = original.cam_quat[cam_id]

    def render_with_trajectory(
        self,
        trajectory_xyz: np.ndarray,
        camera: str = CAMERA_PRIMARY,
    ) -> np.ndarray:
        """Render scene with trajectory markers overlaid.

        Args:
            trajectory_xyz: (N, 3) array of XYZ positions to visualize.
            camera: Camera to render from.

        Returns:
            RGB image with trajectory markers.
        """
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        cam.fixedcamid = cam_id

        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            opt,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )

        # Add trajectory markers as colored spheres
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

        # Render
        self.renderer.update_scene(self.data, camera=camera)
        # Use the scene we built
        ctx = mujoco.gl_context.GLContext(self.width, self.height)
        ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, scene, ctx)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, viewport, ctx)
        img = np.flipud(img)  # OpenGL renders bottom-up
        ctx.free()
        return img

    def close(self) -> None:
        """Clean up renderer resources."""
        self.renderer.close()


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
