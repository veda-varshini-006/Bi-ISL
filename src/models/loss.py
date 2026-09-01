"""Bi-ISL Baseline Training Loss Subsystem (Prompt 27).

Implements:
- Standard sequence Translation Cross-Entropy loss
- Optional Label Smoothing
- Optional Auxiliary CTC Recognition loss
- Optional L2 Weight Regularization
- Configurable loss weighting & ablation controls
- Independent logging for every loss component
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class LossComponents(BaseModel):
    """Schema for independently logged loss components."""

    total_loss: float
    loss_translation: float
    loss_label_smoothed: float = 0.0
    loss_aux_ctc: float = 0.0
    loss_l2_reg: float = 0.0


class BiISLBaselineLoss(nn.Module):
    """Configurable training loss computation module."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        ignore_index: int = 0,
        weight_ctc: float = 0.0,
        weight_reg: float = 0.0,
        ctc_blank_id: int = 0
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.weight_ctc = weight_ctc
        self.weight_reg = weight_reg

        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        self.ctc_criterion = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        encoder_logits: Optional[torch.Tensor] = None,
        ctc_targets: Optional[torch.Tensor] = None,
        ctc_input_lengths: Optional[torch.Tensor] = None,
        ctc_target_lengths: Optional[torch.Tensor] = None,
        model_parameters: Optional[List[nn.Parameter]] = None
    ) -> Tuple[torch.Tensor, LossComponents]:
        """Compute composite loss and return individual component breakdown."""
        B, T_tgt, V = logits.shape

        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        loss_ce = self.ce_criterion(logits_flat, targets_flat)

        loss_ctc = torch.tensor(0.0, device=logits.device)
        if self.weight_ctc > 0.0 and encoder_logits is not None and ctc_targets is not None:
            log_probs = F.log_softmax(encoder_logits, dim=-1).transpose(0, 1)
            loss_ctc = self.ctc_criterion(
                log_probs, ctc_targets, ctc_input_lengths, ctc_target_lengths
            )

        loss_reg = torch.tensor(0.0, device=logits.device)
        if self.weight_reg > 0.0 and model_parameters is not None:
            l2_norm = sum(p.pow(2.0).sum() for p in model_parameters if p.requires_grad)
            loss_reg = 0.5 * l2_norm

        total_loss = loss_ce + (self.weight_ctc * loss_ctc) + (self.weight_reg * loss_reg)

        components = LossComponents(
            total_loss=float(total_loss.item()),
            loss_translation=float(loss_ce.item()),
            loss_label_smoothed=self.label_smoothing,
            loss_aux_ctc=float((self.weight_ctc * loss_ctc).item()),
            loss_l2_reg=float((self.weight_reg * loss_reg).item())
        )

        return total_loss, components
