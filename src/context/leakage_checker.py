"""Automated Context Leakage Checker Subsystem (Prompt 36).

Audits no-context models to guarantee ZERO accidental context leakage:
1. Verifies logits are identical whether context is provided or omitted.
2. Verifies gradients with respect to context inputs are identically zero.
3. Checks configuration flags enforce context.enabled: False.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class ContextLeakageAuditor:
    """Automated auditor verifying zero context leakage in control models."""

    @staticmethod
    def assert_zero_context_leakage(
        control_model: nn.Module,
        sample_batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform 3-point automated zero context leakage check.
        
        Returns audit report dict:
        - bitwise_identical_logits: bool
        - max_logit_difference: float
        - zero_context_gradient: bool
        - config_check_passed: bool
        """
        control_model.eval()

        with torch.no_grad():
            out_no_ctx = control_model(**sample_batch, context_input=None)
            logits_no_ctx = out_no_ctx["logits"] if isinstance(out_no_ctx, dict) else out_no_ctx

        dummy_ctx = torch.randn(logits_no_ctx.shape[0], 256)
        with torch.no_grad():
            out_with_ctx = control_model(**sample_batch, context_input=dummy_ctx)
            logits_with_ctx = out_with_ctx["logits"] if isinstance(out_with_ctx, dict) else out_with_ctx

        max_diff = float(torch.max(torch.abs(logits_no_ctx - logits_with_ctx)).item())
        bitwise_identical = max_diff == 0.0

        if not bitwise_identical:
            raise AssertionError(
                f"Context Leakage Violation Detected! Logit max diff = {max_diff} > 0.0!"
            )

        dummy_ctx_grad = torch.randn(logits_no_ctx.shape[0], 256, requires_grad=True)
        out_grad = control_model(**sample_batch, context_input=dummy_ctx_grad)
        logits_grad = out_grad["logits"] if isinstance(out_grad, dict) else out_grad
        loss = logits_grad.sum()
        loss.backward()

        grad_is_zero = (dummy_ctx_grad.grad is None) or (torch.sum(torch.abs(dummy_ctx_grad.grad)).item() == 0.0)

        if not grad_is_zero:
            raise AssertionError(
                "Context Leakage Violation Detected! Gradients flowed into context_input tensor!"
            )

        return {
            "bitwise_identical_logits": bitwise_identical,
            "max_logit_difference": max_diff,
            "zero_context_gradient": grad_is_zero,
            "config_check_passed": not getattr(control_model, "context_enabled", False),
            "audit_passed": bitwise_identical and grad_is_zero
        }
