"""Evolve Training: M4 Vision & Semantic Module (Stage 5)

Operates at 20 Hz (asynchronous, decoupled from 1000 Hz inner loop).
Performs visual goal identification and 3D pose estimation (Whitepaper §3.5):
T_target_cam, x_d_base = E_v(I_{t-k:t}, instruction)
Forms the full pipeline:
Image -> Object Detection/Pose -> Spatial Kinematics (M2) -> Torque Dynamics (M0) -> Motion
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from evolve_training.core.module_base import EvolveModule


class M4VisionSemanticModule(EvolveModule):
    """Low-frequency (20 Hz) image-to-3D-target pose estimation module."""

    def __init__(
        self,
        in_channels: int = 3,  # RGB (or 4 for RGB-D)
        rate_hz: float = 20.0,
    ):
        super().__init__(module_id="M4_vision_semantic", rate_hz=rate_hz)

        self._input_keys = {"image"}
        self._output_keys = {"x_d_base", "target_confidence"}

        # Visual backbone: lightweight CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # (B, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                                  # (B, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),                           # (B, 128, 4, 4)
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
        )

        # Regressor for 3D target coordinates in Robot Base Frame [x, y, z]
        self.head_pos = nn.Linear(128, 3)
        # Confidence score
        self.head_conf = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        # Nominal table workspace bounds for clamping predictions
        self.register_buffer("ws_min", torch.tensor([0.30, -0.35, 0.25], dtype=torch.float32))
        self.register_buffer("ws_max", torch.tensor([0.75,  0.35, 0.65], dtype=torch.float32))

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Infer target 3D Cartesian coordinates from image."""
        self.validate_inputs(inputs)

        img = inputs["image"]
        if img is None:
            # Fallback when image not captured: default table center
            default_pos = np.array([0.5, 0.0, 0.45], dtype=np.float32)
            outputs = {
                "x_d_base": default_pos,
                "target_confidence": np.float32(0.0),
            }
            return outputs

        if isinstance(img, np.ndarray):
            # Convert (H, W, C) -> (1, C, H, W) in [0, 1]
            img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            img_t = img.float()
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)
            if img_t.shape[1] not in (3, 4):
                img_t = img_t.permute(0, 3, 1, 2)
            if img_t.max() > 1.0:
                img_t = img_t / 255.0

        device = next(self.parameters()).device
        img_t = img_t.to(device)

        features = self.encoder(img_t)
        raw_pos = self.head_pos(features)
        conf = self.head_conf(features)

        # Clamp within valid workspace
        pred_pos = torch.clamp(raw_pos, self.ws_min, self.ws_max)

        if isinstance(inputs["image"], np.ndarray):
            outputs = {
                "x_d_base": pred_pos.squeeze(0).detach().cpu().numpy(),
                "target_confidence": conf.squeeze(0).detach().cpu().numpy(),
            }
        else:
            outputs = {
                "x_d_base": pred_pos,
                "target_confidence": conf,
            }

        self.validate_outputs(outputs)
        return outputs
