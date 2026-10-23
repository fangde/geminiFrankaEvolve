"""Evolve Training: M2 Kinematics Outer Loop Policy (Stage 3)

Operates at 100 Hz.
Maps Cartesian space goals to joint space targets for the 7-DoF redundant Franka arm (Whitepaper §3.3):
q_d, dq_d = f_psi(x_d_base, q_t, dq_t)
Resolves inverse kinematics redundancy, respects joint limits, and outputs smooth setpoints.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from evolve_training.core.module_base import EvolveModule


class M2KinematicsPolicy(EvolveModule):
    """Mid-frequency (100 Hz) spatial goal to joint setpoint mapping policy."""

    def __init__(
        self,
        num_joints: int = 7,
        hidden_dim: int = 256,
        rate_hz: float = 100.0,
    ):
        super().__init__(module_id="M2_kinematics", rate_hz=rate_hz)
        self.num_joints = num_joints

        self._input_keys = {"q", "dq", "x_d_base"}
        self._output_keys = {"q_d", "dq_d", "ddq_d"}

        # Input:
        # q (7), dq (7)
        # x_d_base (7): pos (3) + quat (4)
        # Total = 7 + 7 + 7 = 21 dims
        in_dim = num_joints * 2 + 7

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Output head 1: Delta joint command Delta q_d
        self.head_delta_q = nn.Linear(hidden_dim, num_joints)
        # Output head 2: Desired velocity dq_d
        self.head_dq = nn.Linear(hidden_dim, num_joints)

        # Safe joint boundaries
        self.register_buffer(
            "joint_min",
            torch.tensor([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89], dtype=torch.float32),
        )
        self.register_buffer(
            "joint_max",
            torch.tensor([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89], dtype=torch.float32),
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Infer target joint state q_d, dq_d, ddq_d."""
        self.validate_inputs(inputs)

        q = self._to_tensor(inputs["q"])
        dq = self._to_tensor(inputs["dq"])
        x_d = self._to_tensor(inputs["x_d_base"])

        # Format x_d: if only position (3,), pad with default downward-facing quaternion
        if x_d.shape[-1] == 3:
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x_d.device)
            if x_d.dim() == 1:
                x_d = torch.cat([x_d, default_quat], dim=-1)
            else:
                batch_size = x_d.shape[0]
                quat_batch = default_quat.unsqueeze(0).repeat(batch_size, 1)
                x_d = torch.cat([x_d, quat_batch], dim=-1)

        is_single = (q.dim() == 1)
        if is_single:
            q = q.unsqueeze(0)
            dq = dq.unsqueeze(0)
            x_d = x_d.unsqueeze(0)

        feats = torch.cat([q, dq, x_d], dim=-1)
        hidden = self.net(feats)

        # Predict delta q and desired dq
        delta_q = torch.tanh(self.head_delta_q(hidden)) * 0.2  # max 0.2 rad step per 10ms
        dq_d = torch.tanh(self.head_dq(hidden)) * 2.0         # max 2.0 rad/s

        q_d = torch.clamp(q + delta_q, self.joint_min, self.joint_max)
        ddq_d = torch.zeros_like(dq_d)

        if is_single:
            q_d = q_d.squeeze(0)
            dq_d = dq_d.squeeze(0)
            ddq_d = ddq_d.squeeze(0)

        outputs = {"q_d": q_d, "dq_d": dq_d, "ddq_d": ddq_d}
        self.validate_outputs(outputs)
        return outputs

    def _to_tensor(self, x: Any) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            return x.to(device).float()
        return torch.tensor(x, dtype=torch.float32, device=device)
