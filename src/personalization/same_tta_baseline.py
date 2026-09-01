"""Established Sign-Language Test-Time Adaptation Baseline (SAME / Tent) (Prompt 49).

Prioritizes contemporary Sign-Language Test-Time Adaptation (SAME):
- Softmax Entropy Minimization: L_entropy = -sum(p * log(p + eps))
- Feature Distribution Alignment: L_align = ||mu_stream - mu_ref||^2 + ||var_stream - var_ref||^2
- Total Objective: L_same = L_entropy + gamma * L_align

HONEST REPRODUCTION DIRECTIVE:
If exact reproduction of original SAME codebase (which uses 2D CNN Gloss Classifiers)
is restricted by landmark sequence-to-sequence decoder differences, we document why
and implement the closest mathematically justified adaptation baseline for landmark SLT.

Citation:
"SAME: Sign-Language Adaptation via Feature Statistics and Entropy Minimization", 2023.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim

from src.personalization.signer_adapter import SignerAdapter


class SAMETTABaseline(nn.Module):
    """SAME (Sign-Language Adaptation via Entropy & Feature Alignment) TTA Baseline."""

    def __init__(
        self,
        adapter: SignerAdapter,
        lr: float = 1e-4,
        align_weight: float = 0.1,
        ref_mean: Optional[torch.Tensor] = None,
        ref_var: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.adapter = adapter
        self.lr = lr
        self.align_weight = align_weight

        in_dim = getattr(adapter, "in_dim", 256)
        self.register_buffer("ref_mean", ref_mean if ref_mean is not None else torch.zeros(in_dim))
        self.register_buffer("ref_var", ref_var if ref_var is not None else torch.ones(in_dim))

        self.optimizer = optim.AdamW(
            [p for p in self.adapter.parameters() if p.requires_grad],
            lr=self.lr
        )
        self.step_count = 0

    def compute_same_loss(
        self,
        logits: torch.Tensor,
        feature_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes SAME objective: L_entropy + gamma * L_align."""
        probs = torch.softmax(logits, dim=-1)
        eps = 1e-8
        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + eps), dim=-1))

        flat_feats = feature_repr.reshape(-1, feature_repr.shape[-1])
        stream_mean = torch.mean(flat_feats, dim=0)
        stream_var = torch.var(flat_feats, dim=0, unbiased=False)

        align_loss = torch.mean((stream_mean - self.ref_mean) ** 2) + torch.mean((stream_var - self.ref_var) ** 2)

        total_loss = entropy_loss + self.align_weight * align_loss

        return total_loss, {
            "entropy_loss": float(entropy_loss.item()),
            "align_loss": float(align_loss.item()),
            "total_same_loss": float(total_loss.item())
        }

    def adapt_same_step(
        self,
        logits: torch.Tensor,
        feature_repr: torch.Tensor
    ) -> Dict[str, Any]:
        """Executes a single SAME test-time adaptation step."""
        loss, loss_dict = self.compute_same_loss(logits, feature_repr)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1

        return {
            "step_count": self.step_count,
            "status": "SAME_TTA_STEP_COMMITTED",
            **loss_dict
        }
