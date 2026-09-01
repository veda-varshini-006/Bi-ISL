"""Bi-ISL Learned Visual Encoder RGB/Video Baseline Model (Prompt 23).

Strictly separates:
1. Frame/Video Encoder: Spatial 2D CNN feature extractor (MobileNetV3 / ResNet-18)
2. Temporal Modeling: Multi-layer 1D Temporal Convolutional Network (TCN)
3. Translation Decoder: Autoregressive / sequence projection decoder to target vocabulary

Does NOT integrate SBDS context gating or UGSA signer adaptation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
class FrameEncoder(nn.Module):
    """Spatial 2D CNN frame feature encoder mapping RGB frames (B, T, C, H, W) to (B, T, feature_dim)."""

    def __init__(self, feature_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.pretrained = pretrained

        try:
            import torchvision.models as models
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(576, feature_dim)
            self.use_torchvision = True
        except ImportError:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(128, feature_dim)
            self.use_torchvision = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, T, C, H, W) -> Output: (B, T, feature_dim)."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat)
        pooled = feats.view(B * T, -1)
        out = self.fc(pooled)
        return out.view(B, T, -1)


class TemporalEncoder(nn.Module):
    """Multi-layer 1D Temporal Convolutional Network (TCN) for temporal sequence modeling."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(d_in, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, T, in_dim) -> Output: (B, T, hidden_dim)."""
        x_trans = x.transpose(1, 2)
        out_trans = self.tcn(x_trans)
        return out_trans.transpose(1, 2)


class TranslationDecoder(nn.Module):
    """Autoregressive sequence decoder projecting temporal features to target vocabulary logits."""

    def __init__(self, hidden_dim: int = 512, vocab_size: int = 100, num_layers: int = 1):
        super().__init__()
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, T, hidden_dim) -> Output logits: (B, T, vocab_size)."""
        out, _ = self.decoder_rnn(x)
        logits = self.classifier(out)
        return logits


class RGBVideoBaseline(nn.Module):
    """Complete learned visual encoder RGB/Video baseline architecture."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 100,
        pretrained: bool = False
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(feature_dim=feature_dim, pretrained=pretrained)
        self.temporal_encoder = TemporalEncoder(in_dim=feature_dim, hidden_dim=hidden_dim)
        self.translation_decoder = TranslationDecoder(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(
        self,
        rgb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass taking RGB frames (B, T, C, H, W) -> logits (B, T, vocab_size)."""
        spatial_feats = self.frame_encoder(rgb)
        temporal_feats = self.temporal_encoder(spatial_feats)
        logits = self.translation_decoder(temporal_feats)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(logits)
            logits = logits * mask_expanded.float()

        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown per component and total."""
        f_params = sum(p.numel() for p in self.frame_encoder.parameters() if p.requires_grad)
        t_params = sum(p.numel() for p in self.temporal_encoder.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.translation_decoder.parameters() if p.requires_grad)
        total = f_params + t_params + d_params
        return {
            "frame_encoder_parameters": f_params,
            "temporal_encoder_parameters": t_params,
            "translation_decoder_parameters": d_params,
            "total_parameters": total
        }
