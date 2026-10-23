"""Evolve Training: Joint RL Engine with Retention Loss

Implements PPO fine-tuning with old capability retention:
max_{Theta_k} E[sum gamma^t r_t] - lambda_retain * L_retain - lambda_iface * L_interface
Whitepaper §2.1, §9 & §12.1.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RolloutBuffer:
    """Stores experience tuples for PPO optimization."""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def is_full(self) -> bool:
        return len(self.rewards) >= self.capacity


class JointPPO:
    """PPO implementation supporting multi-module joint optimization and retention loss."""

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 42,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        lambda_retain: float = 0.05,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.lambda_retain = lambda_retain

        # Critic value network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 1),
        ).to(self.device)

        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Log std for continuous Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device) - 0.5)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_val: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        vals = values + [next_val]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * vals[step + 1] * (1.0 - dones[step]) - vals[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + v for adv, v in zip(advantages, values)]
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        return adv_tensor, ret_tensor

    def update_critic(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """Update value function baseline."""
        values = self.critic(states).squeeze(-1)
        loss = nn.functional.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        return loss.item()
