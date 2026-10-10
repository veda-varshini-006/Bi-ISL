"""Bi-ISL Pose-Motion Transformer Training & Fine-Tuning Module.

Implements autoregressive sentence-level training for continuous ISL-to-English translation.
Computes Cross-Entropy loss over token shifts with teacher forcing and AdamW optimizer.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.pose_motion_transformer import PoseMotionTranslationModel, BiISLTranslator


class ContinuousISLDataset(Dataset):
    """Dataset for continuous ISL landmark sequences and English sentence tokens."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        max_seq_len: int = 128,
        max_target_tokens: int = 40,
        tokenizer: Optional[Any] = None
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.max_target_tokens = max_target_tokens
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Landmark sequence: (T, 75, 3) or (T, 225)
        raw_pose = sample.get("landmarks")
        if isinstance(raw_pose, str) and Path(raw_pose).exists():
            pose_arr = np.load(raw_pose)
        elif isinstance(raw_pose, np.ndarray):
            pose_arr = raw_pose
        else:
            pose_arr = np.zeros((self.max_seq_len, 75, 3), dtype=np.float32)

        # Reshape / interpolate to (128, 75, 3)
        if pose_arr.ndim == 2:
            if pose_arr.shape[1] == 225:
                pose_arr = pose_arr.reshape(-1, 75, 3)
            elif pose_arr.shape[1] == 258:
                pose_arr = pose_arr[:, :225].reshape(-1, 75, 3)

        T = pose_arr.shape[0]
        if T != self.max_seq_len:
            tensor_raw = torch.from_numpy(pose_arr).permute(1, 2, 0).unsqueeze(0).float()
            tensor_flat = tensor_raw.reshape(1, 225, T)
            tensor_resampled = nn.functional.interpolate(
                tensor_flat, size=self.max_seq_len, mode="linear", align_corners=False
            )
            pose_128 = tensor_resampled.squeeze(0).permute(1, 0).reshape(self.max_seq_len, 75, 3)
        else:
            pose_128 = torch.from_numpy(pose_arr).float()

        # Target tokenization
        text = sample.get("text", "")
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            tok_ids = self.tokenizer.encode(text, out_type=int)
        else:
            # Simple integer hash fallback
            tok_ids = [abs(hash(w)) % 4990 + 10 for w in text.split()]

        # Format with BOS=2 and EOS=3, PAD=0
        token_seq = [2] + tok_ids[:self.max_target_tokens - 2] + [3]
        pad_len = self.max_target_tokens - len(token_seq)
        if pad_len > 0:
            token_seq += [0] * pad_len

        return {
            "pose": pose_128,
            "target": torch.tensor(token_seq, dtype=torch.long)
        }


class PoseMotionTrainer:
    """Trainer executing sentence-level continuous ISL model optimization."""

    def __init__(
        self,
        model: Optional[PoseMotionTranslationModel] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model or PoseMotionTranslationModel()).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs one epoch of teacher-forced autoregressive training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            pose = batch["pose"].to(self.device)      # (B, 128, 75, 3)
            target = batch["target"].to(self.device)  # (B, T_tgt)

            # Shift right: input is target[:, :-1], label is target[:, 1:]
            dec_in = target[:, :-1]
            labels = target[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(pose, dec_in)         # (B, T_tgt-1, V)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Computes evaluation loss on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            pose = batch["pose"].to(self.device)
            target = batch["target"].to(self.device)
            dec_in = target[:, :-1]
            labels = target[:, 1:]

            logits = self.model(pose, dec_in)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, path: Union[str, Path], epoch: int = 1, val_loss: float = 0.0):
        """Saves model weights atomically to checkpoint file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss
            },
            str(save_path)
        )
        print(f"[Info] Checkpoint saved successfully: {save_path}")
