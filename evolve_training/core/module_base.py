"""Evolve Training: Core Module Base Interface

Defines the base class for all modules M_i in the Evolve Training hierarchy.
Enforces explicit input/output contracts to prevent implicit coupling,
and supports dynamic freezing, parameter tracking, and interface serialization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Set
import torch
import torch.nn as nn


class EvolveModule(nn.Module, ABC):
    """Base class for all hierarchically composable modules in Evolve Training.

    Attributes:
        module_id (str): Identifier, e.g., 'M0_dynamics', 'M1_payload', etc.
        rate_hz (float): Operating frequency in Hz (e.g., 1000.0, 100.0, 20.0).
        is_frozen (bool): Whether the module parameters are frozen during training.
    """

    def __init__(self, module_id: str, rate_hz: float):
        super().__init__()
        self.module_id = module_id
        self.rate_hz = rate_hz
        self.is_frozen = False
        self._input_keys: Set[str] = set()
        self._output_keys: Set[str] = set()

    @property
    def input_keys(self) -> Set[str]:
        """Expected keys in the input dictionary."""
        return self._input_keys

    @property
    def output_keys(self) -> Set[str]:
        """Keys guaranteed in the output dictionary."""
        return self._output_keys

    def freeze(self) -> None:
        """Freeze module parameters so gradients are not computed."""
        self.is_frozen = True
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze module parameters for joint RL or adaptation."""
        self.is_frozen = False
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Verify that all required input keys are present."""
        missing = self._input_keys - set(inputs.keys())
        if missing:
            raise KeyError(
                f"[{self.module_id}] Missing required input keys: {missing}. "
                f"Expected: {self._input_keys}, Provided: {set(inputs.keys())}"
            )

    def validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """Verify that all guaranteed output keys are present."""
        missing = self._output_keys - set(outputs.keys())
        if missing:
            raise KeyError(
                f"[{self.module_id}] Output dictionary missing guaranteed keys: {missing}. "
                f"Guaranteed: {self._output_keys}, Provided: {set(outputs.keys())}"
            )

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass executing module inference.

        Args:
            inputs: Dictionary containing required input tensors.

        Returns:
            Dictionary containing output tensors adhering to output_keys contract.
        """
        pass

    def get_trainable_parameters(self):
        """Return parameters requiring gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def reset_state(self) -> None:
        """Reset internal recurrent states or buffers (if any)."""
        pass
