"""Unit tests for Bi-ISL Landmark Sequence Baseline Models and Trainer."""

import os
import tempfile
import pytest
import torch

from src.models.landmark_baseline import LandmarkSequenceBaseline, LandmarkBaselineTrainer
from src.experiments.e1_landmark_baseline import run_e1_landmark_experiment
from src.data.dataset import SyntheticISLDataset
from src.data.dataloader import create_biisl_dataloader


def test_gru_and_bilstm_forward_pass():
    """Test forward pass for GRU and BiLSTM sequence models."""
    for rnn_type in ["GRU", "BiLSTM"]:
        model = LandmarkSequenceBaseline(input_dim=258, hidden_dim=32, num_layers=1, vocab_size=20, rnn_type=rnn_type)
        dummy_x = torch.randn(2, 16, 258)
        dummy_mask = torch.ones(2, 16, dtype=torch.bool)
        dummy_mask[1, 10:] = False  # Mask padded timesteps

        logits = model(dummy_x, dummy_mask)
        assert logits.shape == (2, 16, 20)


def test_parameter_count_and_latency_measurement():
    """Test parameter counting and latency measurement."""
    model = LandmarkSequenceBaseline(input_dim=258, hidden_dim=64, rnn_type="GRU")
    params = model.count_parameters()
    assert params > 0

    latency_ms = model.measure_inference_latency(batch_size=1, seq_len=16, num_runs=5)
    assert latency_ms > 0.0


def test_landmark_baseline_trainer_and_checkpoint():
    """Test training epoch, validation, and checkpoint saving."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = LandmarkSequenceBaseline(input_dim=258, hidden_dim=16, vocab_size=10, rnn_type="BiLSTM")
        trainer = LandmarkBaselineTrainer(model, lr=1e-3, checkpoint_dir=tmp_dir)

        dataset = SyntheticISLDataset(num_samples=4, modality="landmark", max_seq_len=16)
        loader = create_biisl_dataloader(dataset, batch_size=2, shuffle=False)

        train_loss = trainer.train_epoch(loader)
        val_loss = trainer.validate(loader)

        assert train_loss >= 0.0
        assert val_loss >= 0.0

        ckpt_file = trainer.save_checkpoint(epoch=1, val_loss=val_loss, filename="test_ckpt.pt")
        assert os.path.exists(ckpt_file)


def test_e1_experiment_runner():
    """Test running E1 experiment comparing GRU vs BiLSTM."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_rep, md_rep = run_e1_landmark_experiment(epochs=1, batch_size=2, output_dir=tmp_dir)
        assert os.path.exists(json_rep)
        assert os.path.exists(md_rep)

        with open(md_rep, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Experiment E1: Landmark Sequence Baseline Comparison" in md_text
        assert "GRU" in md_text
        assert "BiLSTM" in md_text
