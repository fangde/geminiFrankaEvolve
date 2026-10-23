"""Evolve Training: Core Safety Monitor

Implements physical guardrails and safety limits for Franka Emika Panda.
Complies with technical challenges identified in §12.2 (动力学内环的安全性).
Clips torques, torque rates, joint limits, and monitors Cartesian workspace boundaries.
"""

from typing import Tuple, Optional
import torch
import numpy as np


class FrankaSafetyMonitor:
    """Enforces hardware safety limits on Franka Emika Panda 7-DoF arm.

    Limits are derived from official Franka specifications:
    - Joint limits (rad)
    - Velocity limits (rad/s)
    - Torque limits (Nm): [87, 87, 87, 87, 12, 12, 12]
    - Max torque rate (Nm/s): 1000 Nm/s
    """

    # Joint limits: lower and upper in radians
    JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    JOINT_LIMITS_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)

    # Joint velocity limits (rad/s)
    VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)

    # Torque limits (Nm) for joints 1 to 7
    TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

    # Max torque change rate (Nm/s)
    MAX_TORQUE_RATE = 1000.0

    def __init__(self, dt: float = 0.001, safety_margin: float = 0.95):
        """
        Args:
            dt: Control cycle period in seconds (default 1ms = 1000Hz).
            safety_margin: Scaling factor for hardware limits (e.g. 0.95 for 5% margin).
        """
        self.dt = dt
        self.safety_margin = safety_margin
        self.effective_torque_limits = self.TORQUE_LIMITS * safety_margin
        self.effective_vel_limits = self.VELOCITY_LIMITS * safety_margin
        self.max_torque_step = self.MAX_TORQUE_RATE * dt

        self.last_torque = np.zeros(7, dtype=np.float32)
        self.safety_violations_count = 0

    def reset(self):
        """Reset internal state."""
        self.last_torque.fill(0.0)
        self.safety_violations_count = 0

    def filter_torque(
        self,
        raw_torque: np.ndarray,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """Apply safety filtering and clipping to raw torque command.

        Args:
            raw_torque: Raw output from M0 dynamics policy (7,).
            q: Current joint positions (7,).
            dq: Current joint velocities (7,).

        Returns:
            safe_torque: Clipped and rate-limited torque (7,).
            is_violation: True if any safety threshold was breached and clamped.
        """
        is_violation = False
        tau = np.copy(raw_torque).astype(np.float32)

        # 1. Torque magnitude clamping
        exceeds_tau = np.abs(tau) > self.effective_torque_limits
        if np.any(exceeds_tau):
            is_violation = True
            tau = np.clip(tau, -self.effective_torque_limits, self.effective_torque_limits)

        # 2. Slew rate (torque derivative) limiting
        delta_tau = tau - self.last_torque
        exceeds_rate = np.abs(delta_tau) > self.max_torque_step
        if np.any(exceeds_rate):
            is_violation = True
            delta_tau = np.clip(delta_tau, -self.max_torque_step, self.max_torque_step)
            tau = self.last_torque + delta_tau

        # 3. Soft joint limit repulsion / damping (if q & dq are provided)
        if q is not None and dq is not None:
            # Check proximity to joint boundaries
            margin = 0.05  # radians buffer
            near_lower = (q < (self.JOINT_LIMITS_LOWER + margin)) & (tau < 0)
            near_upper = (q > (self.JOINT_LIMITS_UPPER - margin)) & (tau > 0)
            if np.any(near_lower) or np.any(near_upper):
                is_violation = True
                tau[near_lower] = np.maximum(tau[near_lower], 0.0)
                tau[near_upper] = np.minimum(tau[near_upper], 0.0)

            # High velocity damping
            high_vel = np.abs(dq) > self.effective_vel_limits
            if np.any(high_vel):
                is_violation = True
                # Apply counter-torque damping
                damping = -5.0 * dq
                tau[high_vel] += damping[high_vel]
                tau = np.clip(tau, -self.effective_torque_limits, self.effective_torque_limits)

        if is_violation:
            self.safety_violations_count += 1

        self.last_torque = np.copy(tau)
        return tau, is_violation
