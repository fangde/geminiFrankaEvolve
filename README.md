# Evolve Training: Franka Emika Panda Implementation

[![Paper Website](https://img.shields.io/badge/Paper-GitHub%20Pages-indigo.svg)](https://fangde.github.io/geminiFrankaEvolve/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/paper_evolve_training.pdf)
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

## 真实实验结果速览 (Empirical Benchmarks)

基于本地严格物理环境评测数据（MuJoCo 真实物理闭环，40 个随机目标，1.0–3.0 kg 随机负载）：
1. **多环行为克隆（BC）的数据扩增与纠偏鲁棒性（Exp 1）**：
   - 原始朴素模仿因严重姿态外推（OOD）导致闭环发散（RMSE 高达 **438.80 mm** / 564.1 mm）；
   - 在全工作空间扩充至 48.5 万样本并注入 DAgger 扰动纠偏样本后，闭环 RMSE 暴降 **99.5% 达到 2.07 mm**，彻底攻克闭环失稳。
2. **端到端联合强化微调的飞跃提升（Exp 2）**：
   - 纯 BC 先验虽可破 1 mm（平均 0.99 mm），但面对复杂工况存在尾部误差；
   - 经过端到端任务损失联合强化（Joint Task FT）微调后，平均跟踪误差收敛至 **0.77 mm（770 微米）**，极端最坏误差仅 **1.54 mm**，到点成功率达到稳固的 **100.0%**。
3. **负载辨识向先验收缩特性的破解（Exp 3）**：
   - 扩充至 5760 个时序窗口后，质量估计 MAE 从 0.247 kg 压降 67.2% 至 **0.081 kg**（仅 81 克误差）。
4. **异步多频与感知延迟边界（Exp 5）**：
   - 1000 Hz 动力学内环提供稳定的阻尼隔离，确保系统在 100–200 ms 严重感知延迟下绝对不发散；但实测显示 100 ms 延迟将末端跟踪误差从 11.3 mm 扩大至 32.7 mm，证实感知延迟是控制精度的首要瓶颈。
