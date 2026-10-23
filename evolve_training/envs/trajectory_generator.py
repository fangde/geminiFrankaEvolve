"""Evolve Training: Trajectory Generator

Generates diverse motion trajectories for Franka Panda:
1. Multi-frequency Fourier series trajectories (for dynamic excitation and M0/M1 training)
2. Fifth-order minimum-jerk spatial trajectories (for M2 kinematics training)
"""

from typing import Tuple, Optional
import numpy as np


class TrajectoryGenerator:
    """Generates smooth, kinematically and dynamically compliant trajectories."""

    def __init__(self, dt: float = 0.001):
        self.dt = dt

    def generate_fourier_trajectory(
        self,
        duration: float = 10.0,
        num_harmonics: int = 5,
        base_freq: float = 0.15,
        q_mean: Optional[np.ndarray] = None,
        amplitude_scale: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate persistent excitation multi-joint Fourier trajectories.

        q_i(t) = q_0,i + sum_{k=1}^K [ a_{i,k} / (k * w0) * sin(k * w0 * t)
                                     - b_{i,k} / (k * w0) * cos(k * w0 * t) ]

        Returns:
            times: (N,)
            q_traj: (N, 7)
            dq_traj: (N, 7)
            ddq_traj: (N, 7)
        """
        N = int(duration / self.dt)
        times = np.linspace(0, duration, N, endpoint=False)

        if q_mean is None:
            q_mean = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

        w0 = 2.0 * np.pi * base_freq
        q_traj = np.tile(q_mean, (N, 1)).astype(np.float32)
        dq_traj = np.zeros((N, 7), dtype=np.float32)
        ddq_traj = np.zeros((N, 7), dtype=np.float32)

        # Joint limits for safety
        limits_low = np.array([-2.5, -1.5, -2.5, -2.8, -2.5, 0.2, -2.5])
        limits_high = np.array([2.5, 1.5, 2.5, -0.3, 2.5, 3.5, 2.5])

        for j in range(7):
            for k in range(1, num_harmonics + 1):
                wk = k * w0
                ak = np.random.uniform(-1.0, 1.0) * amplitude_scale / k
                bk = np.random.uniform(-1.0, 1.0) * amplitude_scale / k

                # Position contribution
                q_traj[:, j] += (ak / wk) * np.sin(wk * times) - (bk / wk) * np.cos(wk * times)
                # Velocity contribution
                dq_traj[:, j] += ak * np.cos(wk * times) + bk * np.sin(wk * times)
                # Acceleration contribution
                ddq_traj[:, j] += -ak * wk * np.sin(wk * times) + bk * wk * np.cos(wk * times)

            # Clip within safe joint limits
            q_traj[:, j] = np.clip(q_traj[:, j], limits_low[j], limits_high[j])

        return times, q_traj, dq_traj, ddq_traj

    def generate_min_jerk_spatial_target(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        duration: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate smooth 5th-order polynomial trajectory between Cartesian points."""
        N = int(duration / self.dt)
        times = np.linspace(0, duration, N, endpoint=False)
        tau = times / duration

        # s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
        ds = (30 * (tau ** 2) - 60 * (tau ** 3) + 30 * (tau ** 4)) / duration

        delta = (goal_pos - start_pos).reshape(1, 3)
        x_traj = start_pos.reshape(1, 3) + s.reshape(-1, 1) * delta
        dx_traj = ds.reshape(-1, 1) * delta

        return times, x_traj.astype(np.float32), dx_traj.astype(np.float32)
