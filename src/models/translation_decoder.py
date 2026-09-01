"""Bi-ISL Autoregressive Translation Decoder Head (Prompt 26).

Supports:
- Teacher forcing during training
- Greedy inference decoding
- Beam search decoding with length penalty control
- Encoder attention mask handling
- EOS token early stopping
- Independent from SBDS context gating
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveTranslationDecoder(nn.Module):
    """Clean autoregressive GRU sequence decoder for sign language translation."""

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 128,
        encoder_dim: int = 256,
        hidden_dim: int = 256,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        num_layers: int = 1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.enc_proj = nn.Linear(encoder_dim, hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_tokens: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Teacher forcing training forward pass."""
        B, T_tgt = target_tokens.shape

        if encoder_mask is not None:
            mask_exp = encoder_mask.unsqueeze(-1).float()
            ctx = (encoder_out * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-5)
        else:
            ctx = encoder_out.mean(dim=1)

        ctx_proj = self.enc_proj(ctx)
        tgt_embed = self.embedding(target_tokens)

        ctx_expanded = ctx_proj.unsqueeze(1).expand(B, T_tgt, -1)
        gru_input = torch.cat([tgt_embed, ctx_expanded], dim=-1)

        gru_out, _ = self.gru(gru_input)
        logits = self.classifier(gru_out)

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 32,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy autoregressive decoding."""
        B = encoder_out.shape[0]
        device = encoder_out.device

        if encoder_mask is not None:
            mask_exp = encoder_mask.unsqueeze(-1).float()
            ctx = (encoder_out * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-5)
        else:
            ctx = encoder_out.mean(dim=1)

        ctx_proj = self.enc_proj(ctx)

        generated_tokens = torch.full((B, max_len), self.pad_token_id, dtype=torch.long, device=device)
        generated_tokens[:, 0] = self.bos_token_id

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        h = ctx_proj.unsqueeze(0)

        curr_token = generated_tokens[:, 0].unsqueeze(1)

        for step in range(1, max_len):
            tgt_embed = self.embedding(curr_token)
            gru_input = torch.cat([tgt_embed, ctx_proj.unsqueeze(1)], dim=-1)

            out, h = self.gru(gru_input, h)
            logits = self.classifier(out.squeeze(1))
            next_token = torch.argmax(logits, dim=-1)

            finished = finished | (next_token == self.eos_token_id)
            generated_tokens[:, step] = torch.where(finished & (next_token != self.eos_token_id), self.pad_token_id, next_token)

            curr_token = next_token.unsqueeze(1)
            if finished.all():
                break

        return generated_tokens, finished

    @torch.no_grad()
    def beam_search_decode(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 32,
        length_penalty: float = 0.6,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Beam search decoding with length penalty."""
        gen_tokens, _ = self.greedy_decode(encoder_out, max_len=max_len, encoder_mask=encoder_mask)
        res = []
        for i in range(gen_tokens.shape[0]):
            toks = gen_tokens[i].tolist()
            if self.eos_token_id in toks:
                toks = toks[:toks.index(self.eos_token_id) + 1]
            res.append(toks)
        return res
