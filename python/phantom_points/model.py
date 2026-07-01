"""PointNet-based realism refiner.

The network follows the PointNet segmentation design (Qi et al., 2017) without the
input/feature T-Nets: LiDAR scans already live in a canonical sensor frame and the
effects we model (range bias, intensity response, dropout) are *not* rotation
invariant, so learned alignment would only hurt.

    per-point features (N, F)
        -> local MLP   F -> 32 -> 64          (shared, BN + ReLU)
        -> global MLP 64 -> 128 -> 256        (shared, BN + ReLU)
        -> max-pool over the sector           (256 global descriptor)
        -> concat [local64, global256]        (320)
        -> head MLP  320 -> 128 -> 64 -> 3    (BN + ReLU except the last layer)

Outputs per point: ``delta_range`` (m), ``delta_intensity`` and ``drop_logit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from .features import num_features


@dataclass
class RefinerConfig:
    in_features: int = field(default_factory=num_features)
    local_dims: List[int] = field(default_factory=lambda: [32, 64])
    global_dims: List[int] = field(default_factory=lambda: [128, 256])
    head_dims: List[int] = field(default_factory=lambda: [128, 64])
    out_dims: int = 3
    max_delta_range: float = 0.5  # metres, applied via tanh
    max_delta_intensity: float = 0.3


def _shared_mlp(dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Conv1d(d_in, d_out, 1), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PointNetRefiner(nn.Module):
    def __init__(self, cfg: RefinerConfig | None = None):
        super().__init__()
        self.cfg = cfg or RefinerConfig()
        c = self.cfg
        self.local_mlp = _shared_mlp([c.in_features] + c.local_dims)
        self.global_mlp = _shared_mlp([c.local_dims[-1]] + c.global_dims)
        head_in = c.local_dims[-1] + c.global_dims[-1]
        self.head = _shared_mlp([head_in] + c.head_dims)
        self.out = nn.Conv1d(c.head_dims[-1], c.out_dims, 1)
        nn.init.zeros_(self.out.weight)  # start as the identity refinement
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        feats: (B, F, N) per-point features for B sectors padded to N points.
        mask:  (B, N) bool, True for real points. Padded points are excluded from max-pool.
        returns raw outputs (B, 3, N): [delta_range_raw, delta_intensity_raw, drop_logit].
        """
        local = self.local_mlp(feats)
        g = self.global_mlp(local)
        if mask is not None:
            g = g.masked_fill(~mask.unsqueeze(1), float("-inf"))
        glob = g.max(dim=2, keepdim=True).values
        glob = torch.nan_to_num(glob, neginf=0.0)  # fully padded sectors
        x = torch.cat([local, glob.expand(-1, -1, local.shape[2])], dim=1)
        return self.out(self.head(x))

    def decode(self, raw: torch.Tensor):
        """Map raw outputs to physical residuals (bounded) and dropout logits."""
        c = self.cfg
        d_range = c.max_delta_range * torch.tanh(raw[:, 0])
        d_int = c.max_delta_intensity * torch.tanh(raw[:, 1])
        return d_range, d_int, raw[:, 2]
