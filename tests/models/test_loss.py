"""Unit tests for Bi-ISL Baseline Training Loss Subsystem."""

import pytest
import torch
import torch.nn as nn

from src.models.loss import BiISLBaselineLoss, LossComponents


def test_translation_cross_entropy_loss():
    """Test translation cross entropy loss computation with label smoothing."""
    loss_fn = BiISLBaselineLoss(label_smoothing=0.1, ignore_index=0)

    logits = torch.randn(2, 5, 20)
    targets = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])

    total_loss, components = loss_fn(logits, targets)
    assert total_loss.item() > 0.0
    assert components.loss_translation > 0.0
    assert components.loss_label_smoothed == 0.1


def test_auxiliary_ctc_loss_and_ablation():
    """Test auxiliary CTC recognition loss computation and ablation."""
    loss_fn_active = BiISLBaselineLoss(weight_ctc=0.5)
    loss_fn_ablated = BiISLBaselineLoss(weight_ctc=0.0)

    logits = torch.randn(2, 5, 20)
    targets = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])

    encoder_logits = torch.randn(2, 10, 20)
    ctc_targets = torch.tensor([[1, 2, 3], [4, 5, 0]])
    in_lens = torch.tensor([10, 10])
    tgt_lens = torch.tensor([3, 2])

    loss1, comp1 = loss_fn_active(
        logits, targets,
        encoder_logits=encoder_logits,
        ctc_targets=ctc_targets,
        ctc_input_lengths=in_lens,
        ctc_target_lengths=tgt_lens
    )
    assert comp1.loss_aux_ctc > 0.0

    loss2, comp2 = loss_fn_ablated(
        logits, targets,
        encoder_logits=encoder_logits,
        ctc_targets=ctc_targets,
        ctc_input_lengths=in_lens,
        ctc_target_lengths=tgt_lens
    )
    assert comp2.loss_aux_ctc == 0.0


def test_l2_regularization_loss():
    """Test L2 weight regularization penalty calculation."""
    loss_fn = BiISLBaselineLoss(weight_reg=0.01)
    linear = nn.Linear(10, 10)

    logits = torch.randn(2, 4, 20)
    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

    total_loss, comp = loss_fn(logits, targets, model_parameters=list(linear.parameters()))
    assert comp.loss_l2_reg > 0.0


def test_independent_loss_component_logging():
    """Test LossComponents logging object structure."""
    comp = LossComponents(
        total_loss=1.234,
        loss_translation=1.0,
        loss_label_smoothed=0.1,
        loss_aux_ctc=0.2,
        loss_l2_reg=0.034
    )
    d = comp.model_dump()
    assert d["total_loss"] == 1.234
    assert d["loss_translation"] == 1.0
    assert d["loss_aux_ctc"] == 0.2


def test_run_loss_ablation_experiment():
    """Test running baseline loss ablation experiment generating report artifacts."""
    import os
    import tempfile
    from src.experiments.e1_loss_ablation import run_loss_ablation_experiment

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_rep, md_rep = run_loss_ablation_experiment(output_dir=tmp_dir)

        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            text = f.read()

        assert "Baseline Loss Function Ablation Report" in text
        assert "STANDARD_CE" in text
        assert "FULL_COMPOSITE" in text
