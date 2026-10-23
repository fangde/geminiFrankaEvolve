"""Evolve Training: Franka MuJoCo Simulation Environment

Provides the core physics environment for Franka Emika Panda 7-DoF arm:
- 1000 Hz direct torque control (1 ms timestep)
- Payload randomization (0 to 3 kg) for Stage 2 (RMA) & Experiment 4
- Camera extrinsic perturbation for Stage 4 (Adaptive Hand-Eye Calibration)
- RGB & Depth rendering for Stage 5 (Vision & Semantics)
- End-effector Cartesian kinematics and Jacobians
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import mujoco


class FrankaBaseEnv:
    """MuJoCo environment for Franka Emika Panda 7-DoF arm."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        dt: float = 0.001,
        render_width: int = 256,
        render_height: int = 256,
    ):
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            xml_path = os.path.join(base_dir, "assets", "franka_emika_panda", "evolve_scene.xml")

        self.xml_path = xml_path
        self.dt = dt
        self.render_width = render_width
        self.render_height = render_height

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        # Lookup IDs
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.joint_qvel_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        self.payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload_body")
        self.overhead_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        self.wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        # Initial home joint configuration
        self.home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

        # Offscreen renderer (lazy initialized)
        self._renderer: Optional[mujoco.Renderer] = None

        # Tracking state
        self.current_payload_mass = 0.0
        self.nominal_overhead_cam_pos = np.copy(self.model.cam_pos[self.overhead_cam_id])
        self.nominal_overhead_cam_quat = np.copy(self.model.cam_quat[self.overhead_cam_id])

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        return self._renderer

    def reset(self, randomize_target: bool = True, randomize_payload: bool = False) -> Dict[str, Any]:
        """Reset environment to home configuration with optional perturbations."""
        mujoco.mj_resetData(self.model, self.data)

        # Set arm home positions
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = self.home_q[i]
            self.data.qvel[self.joint_qvel_adr[i]] = 0.0

        # Set target object position on table
        if randomize_target and self.target_joint_id != -1:
            target_qpos_adr = self.model.jnt_qposadr[self.target_joint_id]
            # Randomize x in [0.45, 0.65], y in [-0.2, 0.2], z on table = 0.42
            tx = np.random.uniform(0.45, 0.65)
            ty = np.random.uniform(-0.20, 0.20)
            self.data.qpos[target_qpos_adr : target_qpos_adr + 3] = [tx, ty, 0.42]
            self.data.qpos[target_qpos_adr + 3 : target_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Reset payload
        if randomize_payload:
            self.set_payload_mass(np.random.uniform(1.0, 3.0))
        else:
            self.set_payload_mass(0.0)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def set_payload_mass(self, mass: float):
        """Dynamically modify end-effector payload mass in kg (Whitepaper §3.2)."""
        self.current_payload_mass = float(mass)
        # Update body mass
        self.model.body_mass[self.payload_body_id] = max(0.001, mass)
        # Scale inertia box accordingly
        r = 0.02
        box_inertia = (1.0 / 6.0) * max(0.001, mass) * (2 * r) ** 2
        self.model.body_inertia[self.payload_body_id] = [box_inertia, box_inertia, box_inertia]
        # Re-derive dependent constants
        mujoco.mj_setConst(self.model, self.data)

    def perturb_camera_extrinsic(self, pos_jitter_std: float = 0.03, rot_jitter_rad: float = 0.05):
        """Inject calibration error to overhead camera for Stage 4 (Adaptive Calibration)."""
        dpos = np.random.normal(0, pos_jitter_std, size=3)
        self.model.cam_pos[self.overhead_cam_id] = self.nominal_overhead_cam_pos + dpos

        # Small rotation perturbation
        angle = np.random.normal(0, rot_jitter_rad)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8
        dq = np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])
        self.model.cam_quat[self.overhead_cam_id] = dq  # approximately perturbed

    def reset_camera_extrinsic(self):
        """Restore nominal camera extrinsics."""
        self.model.cam_pos[self.overhead_cam_id] = np.copy(self.nominal_overhead_cam_pos)
        self.model.cam_quat[self.overhead_cam_id] = np.copy(self.nominal_overhead_cam_quat)

    def step(self, action_tau: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply 7-dim torque command and advance 1000 Hz physics by 1 ms."""
        # Ensure 7-dim
        tau = np.asarray(action_tau, dtype=np.float64).flatten()[:7]
        self.data.ctrl[:7] = tau

        # Physics step (1ms)
        mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        ee_pos = obs["ee_pos"]
        target_pos = obs["target_pos"]
        dist = np.linalg.norm(ee_pos - target_pos)

        # Dense tracking reward
        reward = -dist - 0.01 * np.sum(np.square(tau))
        done = False
        info = {
            "dist": dist,
            "payload_mass": self.current_payload_mass,
            "ee_pos": ee_pos,
            "target_pos": target_pos,
        }
        return obs, reward, done, info

    def get_obs(self) -> Dict[str, Any]:
        """Read proprioceptive and Cartesian sensory states."""
        q = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr], dtype=np.float32)
        dq = np.array([self.data.qvel[adr] for adr in self.joint_qvel_adr], dtype=np.float32)

        # End-effector Cartesian pose
        ee_pos = np.copy(self.data.site_xpos[self.ee_site_id]).astype(np.float32)
        ee_mat = np.copy(self.data.site_xmat[self.ee_site_id]).reshape(3, 3).astype(np.float32)

        # Target Cartesian pose
        target_pos = np.copy(self.data.site_xpos[self.target_site_id]).astype(np.float32)

        return {
            "q": q,
            "dq": dq,
            "ee_pos": ee_pos,
            "ee_mat": ee_mat,
            "target_pos": target_pos,
            "payload_mass": self.current_payload_mass,
        }

    def render_camera(self, camera_name: str = "overhead_cam") -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB and Depth image from specified camera."""
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def get_gravity_compensation_torque(self) -> np.ndarray:
        """Compute analytical gravity compensation torque using MuJoCo inverse dynamics."""
        # Save qacc and qvel
        orig_qacc = np.copy(self.data.qacc)
        orig_qvel = np.copy(self.data.qvel)

        self.data.qacc[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        grav_tau = np.copy(self.data.qfrc_inverse[:7]).astype(np.float32)

        # Restore
        self.data.qacc[:] = orig_qacc
        self.data.qvel[:] = orig_qvel
        return grav_tau

    def get_ee_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector translational and rotational Jacobians."""
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        # Select first 7 arm joints
        return jacp[:, :7].astype(np.float32), jacr[:, :7].astype(np.float32)
