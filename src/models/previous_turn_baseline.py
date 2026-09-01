"""Ungated Previous-Turn Context Baseline Model (Prompt 37).

Represents conventional contextual Sign Language Translation (SLT) baseline:
- Uses previous translated/reference turn text T_{t-1} as unstructured context.
- Encodes T_{t-1} via text embedding + GRU encoder into c_{t-1}.
- Fuses c_{t-1} with visual feature h_t via simple additive projection: h_tilde = h_t + W_{prev}(c_{t-1}).

STRICT RULES:
- Does NOT use SBDS structured state objects.
- Does NOT use reliability gating (alpha_t) or context reliability signals.
- Keeps capacity comparable (~1.2M parameters).
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from src.models.multimodal_baseline import MultimodalBaseline


class PreviousTurnTextEncoder(nn.Module):
    """Text sequence encoder for previous turn translated text."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass converting previous turn tokens (B, L) -> context vector (B, D)."""
        embeds = self.embedding(prev_tokens)  # (B, L, D)
        _, h_n = self.gru(embeds)  # (1, B, D)
        c_prev = self.proj(h_n.squeeze(0))  # (B, D)
        return c_prev


class PreviousTurnBaseline(nn.Module):
    """Conventional contextual SLT baseline fusing ungated previous-turn text context."""

    def __init__(
        self,
        base_visual_model: nn.Module,
        vocab_size: int = 1000,
        embed_dim: int = 256
    ):
        super().__init__()
        self.base_visual_model = base_visual_model
        self.text_encoder = PreviousTurnTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.fuse_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        prev_target_tokens: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Forward pass fusing ungated previous turn text context into visual features."""
        out = self.base_visual_model(**kwargs)
        logits = out["logits"] if isinstance(out, dict) else out

        if prev_target_tokens is not None:
            c_prev = self.text_encoder(prev_target_tokens)
            proj_c = self.fuse_proj(c_prev)
            if logits.dim() == 3 and proj_c.dim() == 2:
                # Expand proj_c to match logits vocabulary dimension if needed
                vocab_dim = logits.shape[-1]
                if proj_c.shape[-1] != vocab_dim:
                    expand_layer = nn.Linear(proj_c.shape[-1], vocab_dim, device=proj_c.device)
                    proj_c = expand_layer(proj_c)
                proj_c = proj_c.unsqueeze(1)
            logits = logits + proj_c

        return {"logits": logits}
