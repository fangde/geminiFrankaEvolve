"""Evolve Training: Three-Step Progressive Training Engine

Implements the 3-step progressive training loop defined in §2.1 & §9:
"Supervised Imitation -> Module Integration -> Joint Reinforcement Learning -> Next Module"

Enforces:
1. Supervised initialization theta_k^(0) = argmin L_sup(M_k, D_k)
2. Module connection with previous modules Theta_{k-1} frozen
3. Joint Reinforcement Learning with retention regularization:
   max_{Theta_k} E[sum gamma^t r_t] - lambda * L_retain
"""

from typing import Dict, Any, List, Optional, Callable
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from evolve_training.core.module_base import EvolveModule


class ProgressiveTrainer:
    """Manages the lifecycle of hierarchical module growth in Evolve Training."""

    def __init__(
        self,
        device: str = "cpu",
        lambda_retain: float = 0.1,
        lambda_interface: float = 0.05,
    ):
        self.device = torch.device(device)
        self.lambda_retain = lambda_retain
        self.lambda_interface = lambda_interface

        # Sequence of registered modules: M0, M1, M2, ...
        self.module_chain: List[EvolveModule] = []
        # Snapshots of frozen parameter states for retention loss calculation
        self.reference_snapshots: Dict[str, Dict[str, torch.Tensor]] = {}

    def register_module(self, module: EvolveModule) -> None:
        """Register a new module into the hierarchy chain."""
        module.to(self.device)
        self.module_chain.append(module)

    # -----------------------------------------------------------------
    # Step 1: Supervised Imitation Learning
    # -----------------------------------------------------------------
    def step1_supervised_imitation(
        self,
        module: EvolveModule,
        dataset: Dataset,
        loss_fn: Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor],
        batch_size: int = 128,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Train module parameters theta_k on heterogeneous task data D_k.

        theta_k^(0) = argmin_{theta_k} L_sup(M_k, D_k)
        """
        module.to(self.device)
        module.unfreeze()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"loss": []}
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                optimizer.zero_grad()
                preds = module(inputs)
                loss = loss_fn(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            history["loss"].append(avg_loss)

        # Snapshot current parameters for future retention loss L_retain
        self._snapshot_module_params(module)
        return history

    # -----------------------------------------------------------------
    # Step 2: Module Integration & Alignment
    # -----------------------------------------------------------------
    def step2_integrate_module(
        self,
        new_module: EvolveModule,
        freeze_previous: bool = True,
    ) -> None:
        """Connect the new module to the existing hierarchy.

        Optionally freezes Theta_{k-1} to avoid destructive interference.
        """
        if freeze_previous:
            for mod in self.module_chain:
                if mod.module_id != new_module.module_id:
                    mod.freeze()

        new_module.unfreeze()
        if new_module not in self.module_chain:
            self.register_module(new_module)

    # -----------------------------------------------------------------
    # Step 3: Joint Reinforcement Learning Optimization
    # -----------------------------------------------------------------
    def compute_retention_loss(self) -> torch.Tensor:
        """Compute parameter distance / behavior anchor retention loss.

        L_retain = sum_{i < k} || theta_i - theta_i_ref ||^2
        """
        loss_retain = torch.tensor(0.0, device=self.device)
        for mod in self.module_chain[:-1]:  # all previous modules
            if mod.module_id in self.reference_snapshots:
                ref_dict = self.reference_snapshots[mod.module_id]
                for name, param in mod.named_parameters():
                    if name in ref_dict:
                        loss_retain += torch.norm(param - ref_dict[name].to(self.device), p=2)
        return loss_retain

    def get_joint_optimizers(
        self,
        lr_new: float = 3e-4,
        lr_old_ratio: float = 0.1,
    ) -> optim.Optimizer:
        """Create optimizer with differential learning rates for joint fine-tuning.

        New module gets full lr_new; unfreezing older modules uses fractional lr.
        """
        param_groups = []
        for i, mod in enumerate(self.module_chain):
            is_new = (i == len(self.module_chain) - 1)
            lr = lr_new if is_new else (lr_new * lr_old_ratio)
            param_groups.append({
                "params": [p for p in mod.parameters() if p.requires_grad],
                "lr": lr,
            })
        return optim.AdamW(param_groups, weight_decay=1e-5)

    def _snapshot_module_params(self, module: EvolveModule):
        """Save a detached clone of module weights."""
        snapshot = {}
        for name, param in module.named_parameters():
            snapshot[name] = param.detach().clone().cpu()
        self.reference_snapshots[module.module_id] = snapshot
