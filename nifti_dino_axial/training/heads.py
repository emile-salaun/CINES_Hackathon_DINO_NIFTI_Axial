"""
DINOv3 projection heads.

Both DINO (CLS-level) and iBOT (patch-level) use the same MLP head architecture:
    Linear → GELU → [hidden layers] → Linear (bottleneck) → L2-norm → weight-norm linear.

This is the standard DINOv2 head, unchanged from FlexiCT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    MLP projection head for DINO and iBOT losses.

    Args:
        in_dim:         Input dimension (backbone embed_dim = 864 for FlexiCT-Base).
        out_dim:        Number of prototypes (65536 in DINOv2 / FlexiCT).
        hidden_dim:     Hidden layer width.
        bottleneck_dim: Width of the L2-normalised bottleneck before the last linear.
        nlayers:        Total number of linear layers (including bottleneck).
        norm_last_layer: Freeze the weight-norm magnitude (first epoch stabilisation).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        assert nlayers >= 2, "Need at least 2 layers (hidden + bottleneck)"

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) — works for both CLS tokens (B, D) and
               patch tokens (B, N, D); the last-layer projection is applied
               element-wise.
        Returns:
            (..., out_dim) raw logits (before softmax).
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
