"""Evolve Training: M3 Adaptive Calibration Encoder (Stage 4)

Operates at 100 Hz.
Estimates extrinsic calibration between robot base and camera/tools (Whitepaper §3.4):
T_hat = E_omega(q_{t-H:t}, x_{t-H:t})
Compensates for sensor mounting errors, camera shifts, and tool length offsets online.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from collections import deque
import numpy as np
from evolve_training.core.module_base import EvolveModule


class M3CalibrationEncoder(EvolveModule):
    """Adaptive Hand-Eye / Tool calibration estimator over motion history."""

    def __init__(
        self,
        history_len: int = 30,  # 30 steps window
        num_joints: int = 7,
        rate_hz: float = 100.0,
    ):
        super().__init__(module_id="M3_calibration", rate_hz=rate_hz)
        self.history_len = history_len
        self.num_joints = num_joints

        self._input_keys = {"q", "dq"}
        self._output_keys = {"T_cam_hat"}

        # Per-step features: joint angles q (7) + external observed ee pose (6: pos 3 + rot 3) = 13 dims
        in_dim = num_joints + 6

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=history_len),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ELU(),
            # Output: 3D position delta (3) + 6D continuous rotation representation (6) = 9 dims
            nn.Linear(64, 9),
        )

        self._history_buffer = deque(maxlen=history_len)

    def reset_state(self) -> None:
        self._history_buffer.clear()

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate 4x4 homogeneous transformation matrix T_cam_hat."""
        if "history" in inputs:
            hist = inputs["history"]
            if isinstance(hist, np.ndarray):
                hist = torch.tensor(hist, dtype=torch.float32)
            if hist.dim() == 2:
                hist = hist.unsqueeze(0)
            if hist.shape[1] == self.history_len:
                hist = hist.transpose(1, 2)  # (B, in_dim, H)
        else:
            self.validate_inputs(inputs)
            q = inputs["q"]
            if isinstance(q, torch.Tensor):
                q = q.detach().cpu().numpy().flatten()
            obs_x = inputs.get("obs_ee_pos", np.zeros(6, dtype=np.float32))
            if isinstance(obs_x, torch.Tensor):
                obs_x = obs_x.detach().cpu().numpy().flatten()
            if len(obs_x) == 3:
                obs_x = np.concatenate([obs_x, np.zeros(3, dtype=np.float32)])

            step_vec = np.concatenate([q, obs_x[:6]])
            self._history_buffer.append(step_vec)
            while len(self._history_buffer) < self.history_len:
                self._history_buffer.append(self._history_buffer[-1])

            arr = np.array(self._history_buffer, dtype=np.float32)
            hist = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

        device = next(self.parameters()).device
        hist = hist.to(device)

        raw_out = self.net(hist)  # (B, 9)
        pos = raw_out[:, :3]
        r6d = raw_out[:, 3:]

        # Convert 6D continuous representation to SO(3) matrix via Gram-Schmidt
        R = self._compute_rotation_matrix_from_ortho6d(r6d)

        # Build homogeneous 4x4 matrix
        B = raw_out.shape[0]
        T_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_mat[:, :3, :3] = R
        T_mat[:, :3, 3] = pos

        if "history" not in inputs:
            T_mat = T_mat.squeeze(0).detach().cpu().numpy()

        outputs = {"T_cam_hat": T_mat}
        self.validate_outputs(outputs)
        return outputs

    def _compute_rotation_matrix_from_ortho6d(self, poses: torch.Tensor) -> torch.Tensor:
        """Zhou et al. Continuous 6D rotation representation."""
        x_raw = poses[:, 0:3]
        y_raw = poses[:, 3:6]

        x = nn.functional.normalize(x_raw, dim=-1)
        z = torch.cross(x, y_raw, dim=-1)
        z = nn.functional.normalize(z, dim=-1)
        y = torch.cross(z, x, dim=-1)

        matrix = torch.stack((x, y, z), dim=-1)
        return matrix
