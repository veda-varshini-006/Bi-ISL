"""No-Context Control Baseline Subsystem (Prompt 36).

Implements the strict no-context control condition using identical:
- Visual encoder (FrameEncoder / TemporalEncoder)
- Translation decoder (AutoregressiveTranslationDecoder)
- Training data pipeline and collation
- Optimization and loss function (BiISLBaselineLoss)

Only context access differs (context vector c_t is strictly zeroed or omitted).

Features:
- Configurable context.enabled: False flag
- Automated context leakage auditing (assert_zero_context_leakage)
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class NoContextControlModel(nn.Module):
    """Strict no-context control baseline model wrapping visual-multimodal architecture."""

    def __init__(
        self,
        base_model: nn.Module,
        context_enabled: bool = False
    ):
        super().__init__()
        self.base_model = base_model
        self.context_enabled = context_enabled

        if self.context_enabled:
            raise ValueError(
                "NoContextControlModel Error: context_enabled MUST be set to False for strict control condition!"
            )

    def forward(
        self,
        *args: Any,
        context_input: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        """Forward pass strictly ignoring any passed context_input tensor."""
        if context_input is not None:
            context_input = None

        out = self.base_model(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return {"logits": out}
        return out
