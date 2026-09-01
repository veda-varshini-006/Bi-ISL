"""Unit tests for Context-Evidence Reliability Gate Subsystem (Prompt 35)."""

import os
import tempfile
import pytest
import torch

from src.context.context_gate import ContextEvidenceGate, GateAblationMode


def test_context_gate_learned_mode():
    """Test forward pass in LEARNED mode computing alpha_t in (0, 1) and h_tilde tensor."""
    gate = ContextEvidenceGate(embed_dim=256, reliability_dim=9, ablation_mode=GateAblationMode.LEARNED)
    h_t = torch.randn(2, 256)
    c_t = torch.randn(2, 256)
    u_t = torch.rand(2, 9)

    h_tilde, alpha_t, diagnostics = gate(h_t, c_t, u_t)

    assert h_tilde.shape == (2, 256)
    assert alpha_t.shape == (2, 1)
    assert torch.all((alpha_t >= 0.0) & (alpha_t <= 1.0))
    assert diagnostics["ablation_mode"] == "LEARNED"


def test_context_gate_ablation_force_zero():
    """Test FORCE_ZERO ablation mode forcing alpha_t = 0.0 and h_tilde == h_t."""
    gate = ContextEvidenceGate(embed_dim=256, reliability_dim=9, ablation_mode=GateAblationMode.FORCE_ZERO)
    h_t = torch.randn(2, 256)
    c_t = torch.randn(2, 256)
    u_t = torch.rand(2, 9)

    h_tilde, alpha_t, diagnostics = gate(h_t, c_t, u_t)

    assert torch.all(alpha_t == 0.0)
    assert torch.allclose(h_tilde, h_t)
    assert diagnostics["ablation_mode"] == "FORCE_ZERO"


def test_context_gate_ablation_force_one():
    """Test FORCE_ONE ablation mode forcing alpha_t = 1.0."""
    gate = ContextEvidenceGate(embed_dim=256, reliability_dim=9, ablation_mode=GateAblationMode.FORCE_ONE)
    h_t = torch.randn(2, 256)
    c_t = torch.randn(2, 256)
    u_t = torch.rand(2, 9)

    h_tilde, alpha_t, diagnostics = gate(h_t, c_t, u_t)

    assert torch.all(alpha_t == 1.0)
    assert diagnostics["ablation_mode"] == "FORCE_ONE"


def test_context_gate_ablation_fixed_constant():
    """Test FIXED_CONSTANT ablation mode forcing alpha_t to fixed constant value."""
    gate = ContextEvidenceGate(embed_dim=256, reliability_dim=9, ablation_mode=GateAblationMode.FIXED_CONSTANT, fixed_alpha_value=0.75)
    h_t = torch.randn(2, 256)
    c_t = torch.randn(2, 256)
    u_t = torch.rand(2, 9)

    h_tilde, alpha_t, diagnostics = gate(h_t, c_t, u_t)

    assert torch.all(alpha_t == 0.75)
    assert diagnostics["ablation_mode"] == "FIXED_CONSTANT"


def test_context_gate_per_sample_alpha_logging():
    """Test per-example inspectability logging of alpha_t to JSONL file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gate = ContextEvidenceGate(embed_dim=256, log_dir=tmp_dir)
        h_t = torch.randn(2, 256)
        c_t = torch.randn(2, 256)
        u_t = torch.rand(2, 9)
        sids = ["sample_gate_1", "sample_gate_2"]

        h_tilde, alpha_t, diagnostics = gate(h_t, c_t, u_t, sample_ids=sids)

        logfile = os.path.join(tmp_dir, "gate_alpha_logs.jsonl")
        assert os.path.exists(logfile)

        with open(logfile, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert "sample_gate_1" in lines[0]
        assert "alpha_t" in lines[0]
