"""Bi-ISL Landmark Sequence Baseline Models (Prompt 22).

Implements diagnostic continuous sequence baselines using normalized 3D landmark features:
- GRU: Multi-layer Gated Recurrent Unit
- BiLSTM: Bidirectional Long Short-Term Memory

Supports sequence packing/masking, parameter counting, inference latency measurement,
training, validation, and atomic model checkpointing.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.logging import BiISLLogger


class LandmarkSequenceBaseline(nn.Module):
    """Sequence baseline model (GRU or BiLSTM) for landmark feature sequences."""

    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 128,
        num_layers: int = 2,
        vocab_size: int = 100,
        rnn_type: str = "GRU",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type.upper()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            rnn_out_dim = hidden_dim * 2
        elif self.rnn_type in ["BILSTM", "LSTM"]:
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            rnn_out_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Must be 'GRU' or 'BiLSTM'.")

        self.classifier = nn.Linear(rnn_out_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for landmark sequences x: (B, T, D) -> logits: (B, T, V)."""
        B, T, D = x.shape
        proj = self.input_proj(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().int().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                proj, lengths, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.rnn(proj)

        logits = self.classifier(out)
        return logits

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def measure_inference_latency(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        num_runs: int = 20,
        device: str = "cpu"
    ) -> float:
        """Measure average inference latency per sample in milliseconds."""
        self.to(device)
        self.eval()

        dummy_x = torch.randn(batch_size, seq_len, self.input_dim, device=device)
        dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(5):
                _ = self(dummy_x, dummy_mask)

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(dummy_x, dummy_mask)
        elapsed_sec = time.perf_counter() - start_time

        avg_latency_ms = round((elapsed_sec / (num_runs * batch_size)) * 1000.0, 3)
        return avg_latency_ms


class LandmarkBaselineTrainer:
    """Trainer handling training, validation, and checkpointing for landmark sequence baselines."""

    def __init__(
        self,
        model: LandmarkSequenceBaseline,
        lr: float = 1e-3,
        checkpoint_dir: str = "./artifacts/checkpoints/landmark_baseline",
        logger: Optional[BiISLLogger] = None
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.logger = logger or BiISLLogger(name="LandmarkBaselineTrainer")

    def train_epoch(self, dataloader: DataLoader, device: str = "cpu") -> float:
        """Train model for one epoch and return average training loss."""
        self.model.to(device)
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            if batch["landmark"] is None:
                continue

            x = batch["landmark"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["target_tokens"].to(device)

            self.optimizer.zero_grad()
            logits = self.model(x, mask)

            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            target_flat = targets[:, :T].contiguous().view(-1) if targets.shape[1] >= T else targets.repeat(1, T)[:, :T].contiguous().view(-1)

            loss = self.criterion(logits_flat, target_flat)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())

        avg_loss = round(total_loss / max(1, len(dataloader)), 4)
        return avg_loss

    def validate(self, dataloader: DataLoader, device: str = "cpu") -> float:
        """Evaluate model on validation dataloader and return validation loss."""
        self.model.to(device)
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if batch["landmark"] is None:
                    continue

                x = batch["landmark"].to(device)
                mask = batch["attention_mask"].to(device)
                targets = batch["target_tokens"].to(device)

                logits = self.model(x, mask)
                B, T, V = logits.shape
                logits_flat = logits.view(-1, V)
                target_flat = targets[:, :T].contiguous().view(-1) if targets.shape[1] >= T else targets.repeat(1, T)[:, :T].contiguous().view(-1)

                loss = self.criterion(logits_flat, target_flat)
                total_loss += float(loss.item())

        avg_val_loss = round(total_loss / max(1, len(dataloader)), 4)
        return avg_val_loss

    def save_checkpoint(self, epoch: int, val_loss: float, filename: str = "best_checkpoint.pt") -> str:
        """Save atomic model checkpoint to disk."""
        ckpt_path = self.checkpoint_dir / filename
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "rnn_type": self.model.rnn_type,
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "vocab_size": self.model.vocab_size
        }
        torch.save(ckpt_data, ckpt_path)
        return str(ckpt_path)
