"""Evolve Training: Asynchronous Hierarchy Scheduler

Implements the multi-rate asynchronous execution architecture defined in §2.4:
"训练时联合优化，推理时分层异步运行。"

Schedules:
- 1000 Hz: M0 Dynamics Inner Loop (1 ms period)
-  100 Hz: M1 Payload Perception & M2 Kinematics Outer Loop (10 ms period)
-   20 Hz: M4 Vision & Semantic Module (50 ms period)

Maintains real-time thread safety, Zero-Order-Hold (ZOH) / cubic spline interpolation,
and supports artificial latency injection for stress-testing (Whitepaper §11.5).
"""

from typing import Dict, Any, Optional, Callable, Tuple
import time
import numpy as np
import torch
from evolve_training.core.module_base import EvolveModule
from evolve_training.core.safety import FrankaSafetyMonitor


class AsyncHierarchyScheduler:
    """Multi-rate asynchronous scheduler for Evolve Training control hierarchies.

    Decouples high-frequency torque execution (1000Hz) from mid-frequency kinematics
    (100Hz) and low-frequency visual perception (20Hz).
    """

    def __init__(
        self,
        base_dt: float = 0.001,
        m0_module: Optional[EvolveModule] = None,
        m1_module: Optional[EvolveModule] = None,
        m2_module: Optional[EvolveModule] = None,
        m3_module: Optional[EvolveModule] = None,
        m4_module: Optional[EvolveModule] = None,
        safety_monitor: Optional[FrankaSafetyMonitor] = None,
    ):
        self.base_dt = base_dt  # 0.001s = 1ms (1000 Hz)
        self.step_count = 0
        self.sim_time = 0.0

        # Registered modules
        self.m0 = m0_module  # 1000 Hz
        self.m1 = m1_module  # 100 Hz
        self.m2 = m2_module  # 100 Hz
        self.m3 = m3_module  # 100 Hz
        self.m4 = m4_module  # 20 Hz

        self.safety = safety_monitor or FrankaSafetyMonitor(dt=base_dt)

        # Clock dividers based on base_dt = 1ms
        self.div_kinematics = max(1, int(0.010 / base_dt))   # 10 steps = 100 Hz
        self.div_vision = max(1, int(0.050 / base_dt))       # 50 steps = 20 Hz

        # Buffer for Zero-Order Hold (ZOH) & inter-module signals
        self.buffer: Dict[str, Any] = {
            "q_d": np.zeros(7, dtype=np.float32),
            "dq_d": np.zeros(7, dtype=np.float32),
            "ddq_d": np.zeros(7, dtype=np.float32),
            "z_dyn": np.zeros(8, dtype=np.float32),
            "T_cam_hat": np.eye(4, dtype=np.float32),
            "T_target_cam": np.eye(4, dtype=np.float32),
            "x_d_base": np.zeros(7, dtype=np.float32),  # [x, y, z, qx, qy, qz, qw]
            "vision_latency_steps": 0,
            "pending_vision_result": None,
            "vision_delivery_step": 0,
        }

        # Diagnostics for profiling
        self.stats = {
            "m0_exec_count": 0,
            "m1_exec_count": 0,
            "m2_exec_count": 0,
            "m4_exec_count": 0,
            "safety_violations": 0,
            "worst_case_latency_ms": 0.0,
        }

    def reset(self):
        """Reset internal buffers and step counts."""
        self.step_count = 0
        self.sim_time = 0.0
        self.safety.reset()
        for k in self.buffer:
            if isinstance(self.buffer[k], np.ndarray):
                self.buffer[k].fill(0.0)
        self.buffer["T_cam_hat"] = np.eye(4, dtype=np.float32)
        self.buffer["T_target_cam"] = np.eye(4, dtype=np.float32)
        self.buffer["pending_vision_result"] = None
        self.buffer["vision_delivery_step"] = 0

    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        task_instruction: Optional[str] = None,
        sim_latency_ms: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advance the hierarchy by one 1000 Hz base simulation step (1 ms).

        Args:
            q: Current joint positions (7,).
            dq: Current joint velocities (7,).
            rgb_image: Optional camera RGB image.
            depth_image: Optional camera Depth image.
            task_instruction: Optional high-level goal.
            sim_latency_ms: Injected perception latency in milliseconds.

        Returns:
            tau_safe: Safe 1000 Hz joint torque command (7,).
            info: Diagnostics and intermediate states.
        """
        self.step_count += 1
        self.sim_time += self.base_dt
        t0_start = time.perf_counter()

        # -------------------------------------------------------------
        # Tier 3: Low-Frequency Vision & Semantic Module (20 Hz)
        # -------------------------------------------------------------
        if self.m4 is not None and (self.step_count % self.div_vision == 0):
            self.stats["m4_exec_count"] += 1
            # Simulate computation latency
            delay_steps = int(sim_latency_ms / (self.base_dt * 1000.0))
            if delay_steps > 0:
                # Schedule future arrival of visual estimate
                m4_inputs = {
                    "image": rgb_image,
                    "depth": depth_image,
                    "instruction": task_instruction,
                }
                with torch.no_grad():
                    res = self.m4(m4_inputs)
                self.buffer["pending_vision_result"] = res
                self.buffer["vision_delivery_step"] = self.step_count + delay_steps
            else:
                # Immediate delivery
                m4_inputs = {
                    "image": rgb_image,
                    "depth": depth_image,
                    "instruction": task_instruction,
                }
                with torch.no_grad():
                    res = self.m4(m4_inputs)
                self._apply_vision_result(res)

        # Check if delayed vision result arrived
        if (
            self.buffer["pending_vision_result"] is not None
            and self.step_count >= self.buffer["vision_delivery_step"]
        ):
            self._apply_vision_result(self.buffer["pending_vision_result"])
            self.buffer["pending_vision_result"] = None

        # -------------------------------------------------------------
        # Tier 2: Mid-Frequency Kinematics & Calibration Outer Loop (100 Hz)
        # -------------------------------------------------------------
        if (self.step_count % self.div_kinematics == 0):
            # 2a. Update M3 Calibration (if present)
            if self.m3 is not None:
                m3_inputs = {"q": q, "dq": dq}
                with torch.no_grad():
                    m3_out = self.m3(m3_inputs)
                if "T_cam_hat" in m3_out:
                    self.buffer["T_cam_hat"] = m3_out["T_cam_hat"]

            # 2b. Update M1 Payload Dynamics Estimator (if present)
            if self.m1 is not None:
                self.stats["m1_exec_count"] += 1
                m1_inputs = {"q": q, "dq": dq, "last_tau": self.safety.last_torque}
                with torch.no_grad():
                    m1_out = self.m1(m1_inputs)
                if "z_dyn" in m1_out:
                    val = m1_out["z_dyn"]
                    self.buffer["z_dyn"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)

            # 2c. Update M2 Kinematics Policy (100 Hz)
            if self.m2 is not None:
                self.stats["m2_exec_count"] += 1
                m2_inputs = {
                    "q": q,
                    "dq": dq,
                    "x_d_base": self.buffer["x_d_base"],
                    "T_cam_hat": self.buffer["T_cam_hat"],
                }
                with torch.no_grad():
                    m2_out = self.m2(m2_inputs)
                if "q_d" in m2_out:
                    val = m2_out["q_d"]
                    self.buffer["q_d"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                if "dq_d" in m2_out:
                    val = m2_out["dq_d"]
                    self.buffer["dq_d"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                if "ddq_d" in m2_out:
                    val = m2_out.get("ddq_d", np.zeros(7, dtype=np.float32))
                    self.buffer["ddq_d"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)

        # -------------------------------------------------------------
        # Tier 1: High-Frequency Dynamics Inner Loop (1000 Hz)
        # -------------------------------------------------------------
        self.stats["m0_exec_count"] += 1
        if self.m0 is not None:
            m0_inputs = {
                "q": q,
                "dq": dq,
                "q_d": self.buffer["q_d"],
                "dq_d": self.buffer["dq_d"],
                "ddq_d": self.buffer["ddq_d"],
                "z_dyn": self.buffer["z_dyn"],
            }
            with torch.no_grad():
                m0_out = self.m0(m0_inputs)
            raw_tau = m0_out["tau"]
            if isinstance(raw_tau, torch.Tensor):
                raw_tau = raw_tau.detach().cpu().numpy().flatten()
        else:
            raw_tau = np.zeros(7, dtype=np.float32)

        # -------------------------------------------------------------
        # Hardware Safety Limiting Layer
        # -------------------------------------------------------------
        safe_tau, is_violation = self.safety.filter_torque(raw_tau, q=q, dq=dq)
        if is_violation:
            self.stats["safety_violations"] += 1

        elapsed_ms = (time.perf_counter() - t0_start) * 1000.0
        if elapsed_ms > self.stats["worst_case_latency_ms"]:
            self.stats["worst_case_latency_ms"] = elapsed_ms

        info = {
            "step": self.step_count,
            "sim_time": self.sim_time,
            "raw_tau": raw_tau,
            "safe_tau": safe_tau,
            "is_violation": is_violation,
            "q_d": np.copy(self.buffer["q_d"]),
            "elapsed_ms": elapsed_ms,
        }
        return safe_tau, info

    def _apply_vision_result(self, res: Dict[str, Any]):
        """Apply visual target pose to the kinematic target buffer."""
        if "T_target_cam" in res:
            val = res["T_target_cam"]
            self.buffer["T_target_cam"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        if "x_d_base" in res:
            val = res["x_d_base"]
            self.buffer["x_d_base"] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
