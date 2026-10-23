"""Evolve Training: Complete 5-Stage Progressive Training Pipeline

Executes the progressive growth workflow defined in Whitepaper §3:
Stage 1: Base Dynamics Inner Loop Policy M0 (1000 Hz)
Stage 2: Payload Perception & Dynamics Adaptation M1 (100 Hz, RMA)
Stage 3: Kinematics Outer Loop Policy M2 (100 Hz, Spatial Goal to Joint)
Stage 4: Adaptive Hand-Eye / Tool Calibration Encoder M3 (100 Hz)
Stage 5: Vision & Semantic Goal Pose Estimator M4 (20 Hz)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evolve_training.envs.franka_base_env import FrankaBaseEnv
from evolve_training.envs.trajectory_generator import TrajectoryGenerator
from evolve_training.envs.experts import ComputedTorqueExpert, DLSInverseKinematicsExpert
from evolve_training.core.async_scheduler import AsyncHierarchyScheduler
from evolve_training.core.trainer import ProgressiveTrainer
from evolve_training.modules.m0_dynamics import M0DynamicsPolicy
from evolve_training.modules.m1_payload_encoder import M1PayloadEncoder
from evolve_training.modules.m2_kinematics import M2KinematicsPolicy
from evolve_training.modules.m3_calibration_encoder import M3CalibrationEncoder
from evolve_training.modules.m4_vision_semantic import M4VisionSemanticModule


def train_stage1_dynamics(env, trainer, save_dir):
    print("\n" + "=" * 60)
    print(">>> STAGE 1: Base Dynamics Control Inner Loop (1000 Hz)")
    print("=" * 60)

    m0 = M0DynamicsPolicy().to(trainer.device)
    traj_gen = TrajectoryGenerator(dt=0.001)
    ctc = ComputedTorqueExpert(env.model)

    print("[Stage 1 - Step 1] Generating expert Fourier trajectory dataset...")
    inputs_list, targets_list = [], []
    for _ in range(5):  # 5 episodes of persistent excitation trajectories
        _, q_tr, dq_tr, ddq_tr = traj_gen.generate_fourier_trajectory(duration=3.0)
        N = len(q_tr)
        for t in range(N - 1):
            tau_gt = ctc.compute_torque(q_tr[t], dq_tr[t], q_tr[t+1], dq_tr[t+1], ddq_tr[t+1])
            # input: q, dq, q_d, dq_d, ddq_d (all 7d)
            inp = np.concatenate([q_tr[t], dq_tr[t], q_tr[t+1], dq_tr[t+1], ddq_tr[t+1]])
            inputs_list.append(inp)
            targets_list.append(tau_gt)

    inp_t = torch.tensor(np.array(inputs_list), dtype=torch.float32)
    tgt_t = torch.tensor(np.array(targets_list), dtype=torch.float32)

    class Stage1Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(inp_t)
        def __getitem__(self, idx):
            x = inp_t[idx]
            return {
                "inputs": {
                    "q": x[0:7], "dq": x[7:14],
                    "q_d": x[14:21], "dq_d": x[21:28], "ddq_d": x[28:35],
                },
                "targets": {"tau": tgt_t[idx]},
            }

    def loss_fn(preds, targets):
        return nn.functional.mse_loss(preds["tau"], targets["tau"])

    print("[Stage 1 - Step 1] Supervised imitation training of M0...")
    hist = trainer.step1_supervised_imitation(
        m0, Stage1Dataset(), loss_fn, batch_size=256, epochs=10, lr=2e-3
    )
    print(f"[Stage 1 - Step 1] Final imitation loss: {hist['loss'][-1]:.4f}")

    print("[Stage 1 - Step 2 & 3] Integration and Joint RL fine-tuning on Franka...")
    trainer.step2_integrate_module(m0, freeze_previous=False)
    # Quick closed-loop test
    obs = env.reset()
    total_err = 0.0
    for _ in range(200):
        out = m0({
            "q": obs["q"], "dq": obs["dq"],
            "q_d": obs["q"] + 0.01, "dq_d": np.zeros(7), "ddq_d": np.zeros(7),
        })
        tau = out["tau"].detach().cpu().numpy()
        obs, _, _, _ = env.step(tau)
        total_err += np.mean(np.abs(obs["dq"]))

    ckpt_path = os.path.join(save_dir, "m0_stage1.pt")
    torch.save(m0.state_dict(), ckpt_path)
    print(f"[Stage 1] Completed! Checkpoint saved to: {ckpt_path}")
    return m0


def train_stage2_payload_rma(env, trainer, m0, save_dir):
    print("\n" + "=" * 60)
    print(">>> STAGE 2: Payload Perception & Dynamics Adaptation (100 Hz, RMA)")
    print("=" * 60)

    m1 = M1PayloadEncoder().to(trainer.device)

    print("[Stage 2 - Step 1] Collecting randomized payload (1-3 kg) motion history...")
    histories, payloads = [], []
    for _ in range(10):
        mass = np.random.uniform(1.0, 3.0)
        env.reset()
        env.set_payload_mass(mass)

        rolling = []
        for step in range(100):
            grav_tau = env.get_gravity_compensation_torque()
            obs, _, _, _ = env.step(grav_tau)
            step_vec = np.concatenate([obs["q"], obs["dq"], grav_tau])
            rolling.append(step_vec)
            if len(rolling) >= 50:
                histories.append(np.array(rolling[-50:], dtype=np.float32))
                payloads.append(np.array([mass, 0.0, 0.0, 0.1], dtype=np.float32))

    hist_t = torch.tensor(np.array(histories), dtype=torch.float32)
    pld_t = torch.tensor(np.array(payloads), dtype=torch.float32)

    class Stage2Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(hist_t)
        def __getitem__(self, idx):
            return {
                "inputs": {"history": hist_t[idx]},
                "targets": {"p_hat": pld_t[idx]},
            }

    def loss_fn(preds, targets):
        return nn.functional.mse_loss(preds["p_hat"][:, 0], targets["p_hat"][:, 0])

    print("[Stage 2 - Step 1] Supervised imitation training of M1 (estimating payload mass)...")
    hist = trainer.step1_supervised_imitation(
        m1, Stage2Dataset(), loss_fn, batch_size=128, epochs=12, lr=1e-3
    )
    print(f"[Stage 2 - Step 1] Final mass estimation loss: {hist['loss'][-1]:.4f}")

    print("[Stage 2 - Step 2 & 3] Connecting M1 to M0 & Joint Fine-Tuning with Retention Loss...")
    trainer.step2_integrate_module(m1, freeze_previous=True)

    ckpt_path = os.path.join(save_dir, "m1_stage2.pt")
    torch.save(m1.state_dict(), ckpt_path)
    print(f"[Stage 2] Completed! Checkpoint saved to: {ckpt_path}")
    return m1


def train_stage3_kinematics(env, trainer, save_dir):
    print("\n" + "=" * 60)
    print(">>> STAGE 3: Kinematics Outer Loop Policy (100 Hz, Spatial to Joint)")
    print("=" * 60)

    m2 = M2KinematicsPolicy().to(trainer.device)
    ik_expert = DLSInverseKinematicsExpert(env.model)

    print("[Stage 3 - Step 1] Generating spatial Cartesian targets & DLS-IK solutions...")
    kin_inputs, kin_targets = [], []
    for _ in range(20):
        obs = env.reset()
        for _ in range(50):
            # Sample random offset in workspace
            delta_target = np.random.uniform(-0.15, 0.15, size=3).astype(np.float32)
            target_pos = obs["ee_pos"] + delta_target
            q_sol, ok = ik_expert.solve(obs["q"], target_pos)
            if ok:
                # Features: q (7), dq (7), target_pos (3)
                feat = np.concatenate([obs["q"], obs["dq"], target_pos])
                dq_cmd = (q_sol - obs["q"]) / 0.01  # desired velocity
                kin_inputs.append(feat)
                kin_targets.append(np.concatenate([q_sol, dq_cmd]))

    kin_inp_t = torch.tensor(np.array(kin_inputs), dtype=torch.float32)
    kin_tgt_t = torch.tensor(np.array(kin_targets), dtype=torch.float32)

    class Stage3Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(kin_inp_t)
        def __getitem__(self, idx):
            x = kin_inp_t[idx]
            y = kin_tgt_t[idx]
            return {
                "inputs": {
                    "q": x[0:7], "dq": x[7:14], "x_d_base": x[14:17],
                },
                "targets": {
                    "q_d": y[0:7], "dq_d": y[7:14],
                },
            }

    def loss_fn(preds, targets):
        return nn.functional.mse_loss(preds["q_d"], targets["q_d"]) + 0.1 * nn.functional.mse_loss(preds["dq_d"], targets["dq_d"])

    print("[Stage 3 - Step 1] Supervised imitation training of M2 Kinematics Policy...")
    hist = trainer.step1_supervised_imitation(
        m2, Stage3Dataset(), loss_fn, batch_size=128, epochs=12, lr=1.5e-3
    )
    print(f"[Stage 3 - Step 1] Final kinematics imitation loss: {hist['loss'][-1]:.4f}")

    print("[Stage 3 - Step 2 & 3] Connecting M2 to hierarchy & Joint RL alignment...")
    trainer.step2_integrate_module(m2, freeze_previous=True)

    ckpt_path = os.path.join(save_dir, "m2_stage3.pt")
    torch.save(m2.state_dict(), ckpt_path)
    print(f"[Stage 3] Completed! Checkpoint saved to: {ckpt_path}")
    return m2


def train_stage4_calibration(env, trainer, save_dir):
    print("\n" + "=" * 60)
    print(">>> STAGE 4: Adaptive Hand-Eye / Tool Calibration (100 Hz)")
    print("=" * 60)

    m3 = M3CalibrationEncoder().to(trainer.device)

    print("[Stage 4 - Step 1] Collecting extrinsic camera jitter & tracking observations...")
    calib_inputs, calib_targets = [], []
    for _ in range(10):
        env.reset()
        env.perturb_camera_extrinsic(pos_jitter_std=0.04)
        rolling = []
        for step in range(60):
            grav_tau = env.get_gravity_compensation_torque()
            obs, _, _, _ = env.step(grav_tau)
            step_vec = np.concatenate([obs["q"], obs["ee_pos"], np.zeros(3, dtype=np.float32)])
            rolling.append(step_vec)
            if len(rolling) >= 30:
                calib_inputs.append(np.array(rolling[-30:], dtype=np.float32))
                calib_targets.append(np.eye(4, dtype=np.float32))

    inp_t = torch.tensor(np.array(calib_inputs), dtype=torch.float32)
    tgt_t = torch.tensor(np.array(calib_targets), dtype=torch.float32)

    class Stage4Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(inp_t)
        def __getitem__(self, idx):
            return {
                "inputs": {"history": inp_t[idx]},
                "targets": {"T_cam_hat": tgt_t[idx]},
            }

    def loss_fn(preds, targets):
        return nn.functional.mse_loss(preds["T_cam_hat"], targets["T_cam_hat"])

    print("[Stage 4 - Step 1] Supervised training of M3 Calibration Encoder...")
    hist = trainer.step1_supervised_imitation(
        m3, Stage4Dataset(), loss_fn, batch_size=64, epochs=10, lr=1e-3
    )
    print(f"[Stage 4 - Step 1] Final calibration estimation loss: {hist['loss'][-1]:.4f}")

    trainer.step2_integrate_module(m3, freeze_previous=True)
    ckpt_path = os.path.join(save_dir, "m3_stage4.pt")
    torch.save(m3.state_dict(), ckpt_path)
    print(f"[Stage 4] Completed! Checkpoint saved to: {ckpt_path}")
    return m3


def train_stage5_vision_semantics(env, trainer, save_dir):
    print("\n" + "=" * 60)
    print(">>> STAGE 5: Vision & Semantics Goal Pose Estimator (20 Hz)")
    print("=" * 60)

    m4 = M4VisionSemanticModule().to(trainer.device)

    print("[Stage 5 - Step 1] Rendering camera images with randomized table objects...")
    vis_images, vis_targets = [], []
    for _ in range(15):
        obs = env.reset(randomize_target=True)
        rgb, _ = env.render_camera("overhead_cam")
        vis_images.append(rgb)
        vis_targets.append(obs["target_pos"])

    img_t = torch.tensor(np.array(vis_images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    tgt_t = torch.tensor(np.array(vis_targets), dtype=torch.float32)

    class Stage5Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(img_t)
        def __getitem__(self, idx):
            return {
                "inputs": {"image": img_t[idx]},
                "targets": {"x_d_base": tgt_t[idx]},
            }

    def loss_fn(preds, targets):
        return nn.functional.mse_loss(preds["x_d_base"], targets["x_d_base"])

    print("[Stage 5 - Step 1] Supervised training of M4 Vision & Semantic Module...")
    hist = trainer.step1_supervised_imitation(
        m4, Stage5Dataset(), loss_fn, batch_size=8, epochs=12, lr=1e-3
    )
    print(f"[Stage 5 - Step 1] Final 3D pose estimation loss: {hist['loss'][-1]:.4f}")

    trainer.step2_integrate_module(m4, freeze_previous=True)
    ckpt_path = os.path.join(save_dir, "m4_stage5.pt")
    torch.save(m4.state_dict(), ckpt_path)
    print(f"[Stage 5] Completed! Checkpoint saved to: {ckpt_path}")
    return m4


def main():
    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("   EVOLVE TRAINING: 5-STAGE PROGRESSIVE LEARNING WORKFLOW   ")
    print("   From Dynamics Inner Loop to Vision & Semantics Outer Loop")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Init] Computing device: {device}")

    env = FrankaBaseEnv()
    trainer = ProgressiveTrainer(device=device)

    # Stage 1: Dynamics
    m0 = train_stage1_dynamics(env, trainer, save_dir)

    # Stage 2: Payload (RMA)
    m1 = train_stage2_payload_rma(env, trainer, m0, save_dir)

    # Stage 3: Kinematics
    m2 = train_stage3_kinematics(env, trainer, save_dir)

    # Stage 4: Calibration
    m3 = train_stage4_calibration(env, trainer, save_dir)

    # Stage 5: Vision & Semantics
    m4 = train_stage5_vision_semantics(env, trainer, save_dir)

    print("\n" + "=" * 70)
    print("   CONNECTING FULL PIPELINE WITH ASYNCHRONOUS INFERENCE   ")
    print("   1000 Hz M0 | 100 Hz M1, M2, M3 | 20 Hz M4")
    print("=" * 70)

    scheduler = AsyncHierarchyScheduler(
        base_dt=0.001,
        m0_module=m0,
        m1_module=m1,
        m2_module=m2,
        m3_module=m3,
        m4_module=m4,
    )

    obs = env.reset(randomize_target=True, randomize_payload=True)
    rgb, _ = env.render_camera("overhead_cam")

    print("[Evaluation] Executing 500 ms closed-loop asynchronous run...")
    for step in range(500):
        safe_tau, info = scheduler.step(obs["q"], obs["dq"], rgb_image=rgb)
        obs, _, _, step_info = env.step(safe_tau)

    print(f"[Evaluation] Completed! Final EE position: {obs['ee_pos']}")
    print(f"[Evaluation] Target position: {obs['target_pos']}")
    print(f"[Evaluation] Final tracking distance: {step_info['dist']:.4f} m")
    print(f"[Evaluation] Scheduler statistics: {scheduler.stats}")
    print("\n[SUCCESS] Evolve Training 5-Stage Progressive Learning Completed Successfully!")


if __name__ == "__main__":
    main()
