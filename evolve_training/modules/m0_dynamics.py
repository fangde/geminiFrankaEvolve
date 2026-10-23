"""Evolve Training: M0 Dynamics Inner Loop Module (Stage 1)

Operates at 1000 Hz.
Directly tied to the robot body and physical actuators (Whitepaper §3.1).
Takes current joint states, desired trajectory, and optional dynamics latent representation,
and predicts direct joint torques:
tau_t = pi_theta(q_t, dq_t, q_d, dq_d, ddq_d, z_dyn)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from evolve_training.core.module_base import EvolveModule


class M0DynamicsPolicy(EvolveModule):
    """High-frequency (1000 Hz) torque-level dynamics tracking policy."""

    def __init__(
        self,
        num_joints: int = 7,
        latent_dim: int = 8,
        hidden_dim: int = 256,
        rate_hz: float = 1000.0,
    ):
        super().__init__(module_id="M0_dynamics", rate_hz=rate_hz)
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        self._input_keys = {"q", "dq", "q_d", "dq_d", "ddq_d"}
        self._output_keys = {"tau"}

        # Input features:
        # e = q_d - q (7)
        # de = dq_d - dq (7)
        # q (7), dq (7), ddq_d (7)
        # z_dyn (latent_dim, default 8)
        # Total = 7 * 5 + 8 = 43 dims
        in_dim = num_joints * 5 + latent_dim

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
            nn.Linear(hidden_dim, num_joints),
        )

        # Scale outputs to Panda torque limits: joints 1-4: 87 Nm, joints 5-7: 12 Nm
        self.register_buffer(
            "torque_scale",
            torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=torch.float32),
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute 1000 Hz torque command."""
        self.validate_inputs(inputs)

        q = self._to_tensor(inputs["q"])
        dq = self._to_tensor(inputs["dq"])
        q_d = self._to_tensor(inputs["q_d"])
        dq_d = self._to_tensor(inputs["dq_d"])
        ddq_d = self._to_tensor(inputs["ddq_d"])

        z_dyn = inputs.get("z_dyn", None)
        if z_dyn is None:
            # Stage 1 default: zero vector
            if q.dim() == 1:
                z_dyn = torch.zeros(self.latent_dim, device=q.device)
            else:
                z_dyn = torch.zeros((q.shape[0], self.latent_dim), device=q.device)
        else:
            z_dyn = self._to_tensor(z_dyn)

        # Ensure batch dimension
        is_single = (q.dim() == 1)
        if is_single:
            q = q.unsqueeze(0)
            dq = dq.unsqueeze(0)
            q_d = q_d.unsqueeze(0)
            dq_d = dq_d.unsqueeze(0)
            ddq_d = ddq_d.unsqueeze(0)
            z_dyn = z_dyn.unsqueeze(0)

        # Feature engineering: tracking errors + raw states
        e = q_d - q
        de = dq_d - dq
        features = torch.cat([e, de, q, dq, ddq_d, z_dyn], dim=-1)

        raw_out = self.net(features)
        # Apply tanh and scale to joint limits
        scaled_tau = torch.tanh(raw_out) * self.torque_scale.to(raw_out.device)

        if is_single:
            scaled_tau = scaled_tau.squeeze(0)

        outputs = {"tau": scaled_tau}
        self.validate_outputs(outputs)
        return outputs

    def _to_tensor(self, x: Any) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            return x.to(device).float()
        return torch.tensor(x, dtype=torch.float32, device=device)
