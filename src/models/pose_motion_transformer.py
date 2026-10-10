"""Bi-ISL Pose-Motion Transformer Translation Architecture.

Processes continuous 3D sign pose sequences [T, 75, 3] with temporal motion delta features [T, 75*6]
and autoregressively decodes natural English sentences via SentencePiece tokenizer (vocab size 5000).
"""

import os
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class PoseMotionEncoder(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(75 * 6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        # pose: (B, T, 75, 3) or (B, T, 225)
        if pose.dim() == 4:
            motion = torch.zeros_like(pose)
            motion[:, 1:] = pose[:, 1:] - pose[:, :-1]
            x = torch.cat([pose, motion], dim=-1).reshape(
                pose.size(0), pose.size(1), -1
            )
        elif pose.dim() == 3 and pose.size(-1) == 225:
            pose_3d = pose.reshape(pose.size(0), pose.size(1), 75, 3)
            motion = torch.zeros_like(pose_3d)
            motion[:, 1:] = pose_3d[:, 1:] - pose_3d[:, :-1]
            x = torch.cat([pose_3d, motion], dim=-1).reshape(
                pose.size(0), pose.size(1), -1
            )
        elif pose.dim() == 3 and pose.size(-1) == 75 * 6:
            x = pose
        else:
            raise ValueError(f"Unexpected pose shape: {pose.shape}")

        x = self.input_projection(x)
        return self.temporal_encoder(x)


class PoseMotionTranslationModel(nn.Module):
    def __init__(self, vocab_size: int = 5000, d_model: int = 256, max_len: int = 40):
        super().__init__()
        self.encoder = PoseMotionEncoder(d_model, 0.1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=4
        )
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, pose: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(pose)
        tgt = self.token_embedding(decoder_input)
        tgt = tgt + self.pos_embedding[:, :tgt.size(1)]
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(
                tgt_len, tgt_len,
                device=tgt.device, dtype=torch.bool
            ),
            diagonal=1
        )
        decoded = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=causal_mask
        )
        return self.output_projection(decoded)


class BiISLTranslator:
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = PoseMotionTranslationModel().to(self.device)

        if checkpoint_path and Path(checkpoint_path).is_file():
            try:
                checkpoint = torch.load(
                    str(checkpoint_path), map_location=self.device,
                    weights_only=False
                )
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"[Warning] Could not load checkpoint from {checkpoint_path}: {e}")

        self.model.eval()

        self.sp = None
        if tokenizer_path and Path(tokenizer_path).is_file() and spm is not None:
            try:
                self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            except Exception as e:
                print(f"[Warning] Could not load tokenizer from {tokenizer_path}: {e}")

    @classmethod
    def load_default(cls, device: Optional[Union[str, torch.device]] = None) -> "BiISLTranslator":
        """Loads default checkpoint and tokenizer from artifacts/checkpoints/ if present."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        ckpt = base_dir / "artifacts" / "checkpoints" / "PositionMotionTransformer_finetune_lr1e5_latest.pt"
        sp_model = base_dir / "artifacts" / "checkpoints" / "BiISL_SentencePiece_5000.model"
        
        # Fallback search paths
        if not ckpt.is_file():
            alt_ckpt = Path(r"C:\Users\ADMIN\Downloads\PositionMotionTransformer_finetune_lr1e5_latest.pt")
            if alt_ckpt.is_file():
                ckpt = alt_ckpt
        if not sp_model.is_file():
            alt_sp = Path(r"C:\Users\ADMIN\Downloads\BiISL_SentencePiece_5000.model")
            if alt_sp.is_file():
                sp_model = alt_sp

        return cls(
            checkpoint_path=ckpt if ckpt.is_file() else None,
            tokenizer_path=sp_model if sp_model.is_file() else None,
            device=device
        )

    @torch.no_grad()
    def translate(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        max_new_tokens: int = 39,
        return_confidence: bool = False
    ) -> Union[str, tuple]:
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        pose = pose.to(dtype=torch.float32)

        if pose.dim() == 2:
            # e.g., (128, 225) -> reshape to (128, 75, 3)
            if pose.size(1) == 225:
                pose = pose.reshape(pose.size(0), 75, 3)
            elif pose.size(1) == 258:
                # 258 landmarks -> slice top 75 (225 values)
                pose = pose[:, :225].reshape(pose.size(0), 75, 3)

        if pose.dim() == 3 and pose.shape != (128, 75, 3):
            # Pad or interpolate to 128 frames if needed
            T = pose.size(0)
            if T != 128 and pose.size(1) == 75 and pose.size(2) == 3:
                # Interpolate temporal dimension
                pose_t = pose.permute(1, 2, 0).unsqueeze(0)  # (1, 75, 3, T)
                pose_t = pose_t.reshape(1, 225, T)
                pose_t = torch.nn.functional.interpolate(pose_t, size=128, mode="linear", align_corners=False)
                pose = pose_t.squeeze(0).permute(1, 0).reshape(128, 75, 3)

        pose_in = pose.unsqueeze(0).to(self.device)
        generated = torch.tensor([[2]], dtype=torch.long, device=self.device)  # BOS token ID 2
        token_probs: List[float] = []

        for _ in range(max_new_tokens):
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda")
            ):
                logits = self.model(pose_in, generated)

            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(probs.argmax(dim=-1).item())
            prob_val = float(probs[0, next_id].item())
            token_probs.append(prob_val)

            if next_id in (0, 3):  # PAD or EOS token ID
                break
            generated = torch.cat([
                generated,
                torch.tensor([[next_id]], device=self.device)
            ], dim=1)

        ids = generated[0, 1:].cpu().tolist()
        if self.sp is not None:
            text = self.sp.decode(ids).strip()
        else:
            text = f"Translated Token IDs: {ids}"

        avg_conf = float(np.mean(token_probs)) if token_probs else 0.0

        if return_confidence:
            return text, avg_conf
        return text
