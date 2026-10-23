# Evolve Training: Franka Emika Panda Implementation

[![Paper Website](https://img.shields.io/badge/Paper-GitHub%20Pages-indigo.svg)](https://fangde.github.io/geminiFrankaEvolve/)
[![Version](https://img.shields.io/badge/version-1.4-blue.svg)](docs/whitepaper_v1.4.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.0-orange.svg)](https://mujoco.org)

> 🌐 **在线论文与交互式实验报告网站已发布**：[https://fangde.github.io/geminiFrankaEvolve/](https://fangde.github.io/geminiFrankaEvolve/)

Implementation of **Evolve Training: Progressive and Composable Learning of Embodied Control Hierarchies (具身智能控制层级的渐进式与可组合学习)** on Franka Emika Panda 7-DoF robotic arm.

---

## 架构概览

```
               [ 20 Hz ]
      M4: 视觉与语义感知模块 (Image -> 3D Target Pose)
               │ (异步缓冲区 / 零阶保持)
               ▼
               [ 100 Hz ]
      M3: 自适应手眼/工具标定  ──►  M2: 运动学空间外环
      (Extrinsic Correction)         (f_psi(x_d) -> q_d, dq_d)
                                               │
                                               ▼
                                      [ 1000 Hz ]
      M1: RMA 负载与动力学感知 ──►  M0: 基础动力学内环策略
      (E_phi(h_t) -> z_dyn)          (pi_theta -> tau)
                                               │
                                               ▼
                                   FrankaSafetyMonitor (限幅保护)
                                               │
                                               ▼
                                    Franka 机械臂本体 (MuJoCo)
```

---

## 核心模块与文件清单

| 层次 | 模块标识 | 运行频率 | 核心职责 | 代码文件 |
|---|---|---|---|---|
| **M0** | 基础动力学内环 | 1000 Hz | 读取关节状态与目标轨迹，输出直接关节控制力矩 $\tau$ | [`m0_dynamics.py`](evolve_training/modules/m0_dynamics.py) |
| **M1** | 负载与动力学感知 | 100 Hz | 基于本体运动历史 $h_t$，在线估计末端负载与惯量表征 $z_{\text{dyn}}$ | [`m1_payload_encoder.py`](evolve_training/modules/m1_payload_encoder.py) |
| **M2** | 运动学空间外环 | 100 Hz | 3D 末端空间目标到 7 自由度关节空间的目标映射与冗余解算 | [`m2_kinematics.py`](evolve_training/modules/m2_kinematics.py) |
| **M3** | 自适应手眼标定 | 100 Hz | 根据运动历史与外部观测在线补偿相机外参漂移与工具偏置 | [`m3_calibration_encoder.py`](evolve_training/modules/m3_calibration_encoder.py) |
| **M4** | 视觉与语义感知 | 20 Hz | 从相机图像中识别目标并回归 3D 空间坐标 | [`m4_vision_semantic.py`](evolve_training/modules/m4_vision_semantic.py) |

---

## 运行与验证

### 1. 运行五阶段渐进式全流程训练
```bash
python scripts/train_stage1_to_5.py
```
训练产物将自动保存至 `checkpoints/`：
- `m0_stage1.pt`
- `m1_stage2.pt`
- `m2_stage3.pt`
- `m3_stage4.pt`
- `m4_stage5.pt`

### 2. 运行验证实验
- **实验 1（渐进式训练与旧能力保留 §11.1）**：
  ```bash
  python scripts/exp1_progressive_vs_e2e.py
  ```
- **实验 3（模块热替换与跨具身/工具能力迁移 §11.3）**：
  ```bash
  python scripts/exp3_module_swap_transfer.py
  ```
- **实验 5（分层多频异步推理基准 §11.5）**：
  ```bash
  python scripts/exp5_async_inference.py
  ```

---

## 实验结果速览

根据五组实验的基准测试数据（见 `docs/`）：
1. **样本效率与旧能力留存（Exp 1）**：
   - Evolve Training 阶段性强化收敛 Loss 为 `0.677`，对比 Monolithic 端到端（`631.8`），训练难度显著降低。
   - 旧能力保留率达到 **96.8%**（对照 Monolithic 全网微调的 52.4% 灾难性遗忘）。
2. **能力热插拔与迁移成本（Exp 3）**：
   - **换相机外参**：仅需微调 M3，迁移样本成本为从头训练的 **3.5%**（节约 96.5% 样本）。
   - **换操作工具**：仅需微调 M1，迁移样本成本为从头训练的 **5.5%**（节约 94.5% 样本）。
   - **换机器人身体**：仅需替换绑定的 M0 动力学内环，保留高层 M2/M4，样本成本仅需 **12.0%**。
3. **异步多频鲁棒性（Exp 5）**：
   - 面对 50 ms–100 ms 视觉感知延迟，1000 Hz 异步动力学内环将末端闭环误差控制在 0.85–0.90 m 内，而同步 20 Hz 策略由于延迟累积导致超调发散至 2.76–4.16 m。
