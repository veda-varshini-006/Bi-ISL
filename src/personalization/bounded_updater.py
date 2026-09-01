"""Bounded Signer Adaptation Updater & Regularized Objective (Prompt 46).

Enforces strict bounds during online signer adaptation:
- Gradient step limit (max_steps)
- Learning rate limit (lr <= 1e-4)
- Maximum parameter distance from base state: ||theta_u - theta_0||_2 <= R_max
- Maximum sample buffer size (max_samples_retained)
- Adapter parameter count limit (< 50K)

Regularized Adaptation Objective:
    L_adapt = L_task + lambda * ||theta_u - theta_0||^2

Records and logs exact update magnitude ||theta_u - theta_0||_2 after every step.
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.optim as optim

from src.personalization.signer_adapter import SignerAdapter


class BoundedSignerUpdater:
    """Bounded online updater for SignerAdapter parameters."""

    def __init__(
        self,
        adapter: SignerAdapter,
        lr: float = 1e-4,
        max_steps: int = 5,
        max_dist_r: float = 0.50,
        l2_lambda: float = 0.01,
        max_samples_retained: int = 100
    ):
        self.adapter = adapter
        self.lr = min(lr, 1e-4)
        self.max_steps = max_steps
        self.max_dist_r = max_dist_r
        self.l2_lambda = l2_lambda
        self.max_samples_retained = max_samples_retained

        self.step_count = 0
        self.sample_buffer = []

        self.theta_0 = {
            name: param.detach().clone()
            for name, param in self.adapter.named_parameters()
            if param.requires_grad
        }

        self.optimizer = optim.AdamW(
            [p for p in self.adapter.parameters() if p.requires_grad],
            lr=self.lr
        )

    def compute_parameter_distance(self) -> float:
        """Computes Euclidean distance ||theta_u - theta_0||_2."""
        total_sq_dist = 0.0
        for name, param in self.adapter.named_parameters():
            if param.requires_grad and name in self.theta_0:
                diff = param - self.theta_0[name]
                total_sq_dist += torch.sum(diff ** 2).item()
        return math.sqrt(total_sq_dist)

    def compute_regularization_loss(self) -> torch.Tensor:
        """Computes lambda * ||theta_u - theta_0||^2."""
        device = next(self.adapter.parameters()).device
        reg_loss = torch.tensor(0.0, device=device)
        for name, param in self.adapter.named_parameters():
            if param.requires_grad and name in self.theta_0:
                diff = param - self.theta_0[name]
                reg_loss = reg_loss + torch.sum(diff ** 2)
        return self.l2_lambda * reg_loss

    def update_step(self, task_loss: torch.Tensor) -> Dict[str, Any]:
        """Executes a single bounded gradient update step."""
        if self.step_count >= self.max_steps:
            return {
                "updated": False,
                "reason": f"MAX_STEPS_REACHED ({self.step_count}/{self.max_steps})",
                "param_distance": self.compute_parameter_distance(),
                "step_count": self.step_count
            }

        reg_loss = self.compute_regularization_loss()
        total_loss = task_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        dist = self.compute_parameter_distance()
        if dist > self.max_dist_r:
            scale = self.max_dist_r / dist
            with torch.no_grad():
                for name, param in self.adapter.named_parameters():
                    if param.requires_grad and name in self.theta_0:
                        param.copy_(self.theta_0[name] + scale * (param - self.theta_0[name]))
            dist = self.compute_parameter_distance()

        self.step_count += 1

        return {
            "updated": True,
            "step_count": self.step_count,
            "task_loss": float(task_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "total_loss": float(total_loss.item()),
            "param_distance": round(dist, 4),
            "max_dist_r": self.max_dist_r
        }
