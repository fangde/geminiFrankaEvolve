"""Evolve Training: Expert Controllers & Solvers

Generates expert supervision data D_k for Step 1 (Supervised Imitation):
1. Computed Torque Controller (CTC) using MuJoCo full inverse dynamics:
   tau = M(q) (ddq_d + Kp * e + Kd * de) + C(q, dq) * dq + g(q)
2. Damped Least-Squares (DLS) Inverse Kinematics solver for M2 Kinematics outer loop:
   dq = J^T (J J^T + lambda^2 I)^(-1) dx
"""

from typing import Tuple, Optional
import numpy as np
import mujoco


class ComputedTorqueExpert:
    """Analytical Computed Torque Controller for Franka Emika Panda."""

    def __init__(
        self,
        model: mujoco.MjModel,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.data = mujoco.MjData(model)

        # Default PD feedback gains
        self.kp = kp if kp is not None else np.array([40.0, 40.0, 40.0, 40.0, 15.0, 10.0, 5.0], dtype=np.float32)
        self.kd = kd if kd is not None else np.array([4.0, 4.0, 4.0, 4.0, 2.0, 1.5, 1.0], dtype=np.float32)

        self.limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_d: np.ndarray,
        dq_d: np.ndarray,
        ddq_d: np.ndarray,
    ) -> np.ndarray:
        """Calculate required joint torques to track reference trajectory."""
        e = q_d - q
        de = dq_d - dq
        a_cmd = ddq_d + self.kp * e + self.kd * de

        self.data.qpos[:7] = q
        self.data.qvel[:7] = dq
        self.data.qacc[:7] = a_cmd

        mujoco.mj_inverse(self.model, self.data)
        tau_id = np.copy(self.data.qfrc_inverse[:7]).astype(np.float32)
        tau_id = np.clip(tau_id, -self.limits, self.limits)
        return tau_id


class DLSInverseKinematicsExpert:
    """Damped Least-Squares (DLS) Jacobian IK solver for Franka 7-DoF arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        damping: float = 0.05,
        step_size: float = 0.2,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.damping = damping
        self.step_size = step_size
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    def solve(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        max_iters: int = 15,
        pos_tolerance: float = 0.005,
    ) -> Tuple[np.ndarray, bool]:
        """Iteratively solve for target joint position q_d matching Cartesian target_pos."""
        q = np.copy(current_q).astype(np.float64)
        success = False

        for _ in range(max_iters):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)

            current_ee_pos = self.data.site_xpos[self.ee_site_id]
            err = target_pos - current_ee_pos
            err_norm = np.linalg.norm(err)

            if err_norm < pos_tolerance:
                success = True
                break

            # Compute Jacobian
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J = jacp[:, :7]  # (3, 7)

            # DLS formula: dq = J^T (J J^T + lambda^2 I)^(-1) err
            JJT = J @ J.T + (self.damping ** 2) * np.eye(3)
            inv_JJT = np.linalg.inv(JJT)
            dq = J.T @ (inv_JJT @ err)

            q += self.step_size * dq

            # Enforce limits
            q = np.clip(
                q,
                [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8],
                [ 2.8,  1.7,  2.8, -0.1,  2.8, 3.7,  2.8],
            )

        return q.astype(np.float32), success
