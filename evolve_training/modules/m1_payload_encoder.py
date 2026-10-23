"""Evolve Training: M1 Payload Perception & Dynamics Adaptation (Stage 2)

Operates at 100 Hz.
Implements Rapid Motor Adaptation (RMA) encoder over proprioceptive history (Whitepaper §3.2 & §7.1):
h_t = {q, dq, tau}_{t-H:t}
p_hat, z_dyn = E_phi(h_t)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from collections import deque
import numpy as np
from evolve_training.core.module_base import EvolveModule


class M1PayloadEncoder(EvolveModule):
    """Temporal 1D-CNN / MLP encoder for online payload and inertia adaptation."""

    def __init__(
        self,
        history_len: int = 50,  # 50 ms history window at 1000 Hz
        num_joints: int = 7,
        latent_dim: int = 8,
        rate_hz: float = 100.0,
    ):
        super().__init__(module_id="M1_payload", rate_hz=rate_hz)
        self.history_len = history_len
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        self._input_keys = {"q", "dq", "last_tau"}
        self._output_keys = {"z_dyn", "p_hat"}

        # Per-step features: q (7), dq (7), tau (7) = 21 dims
        step_dim = num_joints * 3

        # Temporal 1D Convolution over history: (B, 21, 50) -> (B, 32, 25) -> (B, 64, 13)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=step_dim, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 13, 128),
            nn.ELU(),
        )

        # Head 1: Latent dynamics representation z_dyn (to feed M0)
        self.head_latent = nn.Linear(128, latent_dim)

        # Head 2: Explicit physical parameter estimation p_hat = [mass (kg), com_x, com_y, com_z]
        self.head_explicit = nn.Linear(128, 4)

        # Runtime rolling history buffer
        self._history_buffer = deque(maxlen=history_len)

    def reset_state(self) -> None:
        """Clear rolling history."""
        self._history_buffer.clear()

    def update_history(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray):
        """Append one step of (q, dq, tau) to internal rolling buffer."""
        step_vec = np.concatenate([q.flatten(), dq.flatten(), tau.flatten()]).astype(np.float32)
        self._history_buffer.append(step_vec)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Infer payload parameters and latent dynamics vector."""
        if "history" in inputs:
            # Batch mode training with pre-packaged history tensor (B, H, 21) or (B, 21, H)
            hist = inputs["history"]
            if isinstance(hist, np.ndarray):
                hist = torch.tensor(hist, dtype=torch.float32)
            if hist.dim() == 2:
                hist = hist.unsqueeze(0)
            if hist.shape[1] == self.history_len:
                hist = hist.transpose(1, 2)  # (B, 21, H)
        else:
            # Online inference mode using internal buffer
            self.validate_inputs(inputs)
            q = inputs["q"]
            dq = inputs["dq"]
            tau = inputs["last_tau"]
            if isinstance(q, torch.Tensor):
                q = q.detach().cpu().numpy()
            if isinstance(dq, torch.Tensor):
                dq = dq.detach().cpu().numpy()
            if isinstance(tau, torch.Tensor):
                tau = tau.detach().cpu().numpy()

            self.update_history(q, dq, tau)

            # Pad buffer if not full yet
            while len(self._history_buffer) < self.history_len:
                self._history_buffer.append(self._history_buffer[-1])

            arr = np.array(self._history_buffer, dtype=np.float32)  # (H, 21)
            hist = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).transpose(1, 2)  # (1, 21, H)

        device = next(self.parameters()).device
        hist = hist.to(device)

        features = self.conv_net(hist)
        z_dyn = self.head_latent(features)
        p_hat = self.head_explicit(features)

        if "history" not in inputs:
            z_dyn = z_dyn.squeeze(0)
            p_hat = p_hat.squeeze(0)

        outputs = {"z_dyn": z_dyn, "p_hat": p_hat}
        self.validate_outputs(outputs)
        return outputs
