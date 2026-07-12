"""Export a trained refiner to the flat ``.ppnt`` format read by the C++ generator.

Binary layout (little-endian)::

    char[4]  magic "PPNT"
    uint32   version (1)
    uint32   in_features
    uint32   num_classes
    uint32   sectors
    float32  feature_scale
    float32  max_delta_range
    float32  max_delta_intensity
    uint32   n_local, n_global, n_head        # layer counts per block (head includes output)
    repeat for each layer (local, then global, then head):
        uint32  in_dim
        uint32  out_dim
        uint32  relu (0/1)
        float32 weight[out_dim * in_dim]      # row-major, BatchNorm folded in
        float32 bias[out_dim]

``python -m phantom_points.export --checkpoint runs/refiner/best.pt --out refiner.ppnt``
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from . import NUM_CLASSES
from .features import DEFAULT_FEATURE_SCALE, DEFAULT_SECTORS
from .model import PointNetRefiner, RefinerConfig

MAGIC = b"PPNT"
VERSION = 1

Layer = Tuple[np.ndarray, np.ndarray, bool]  # (W[out, in], b[out], relu)


def _fold(conv: nn.Conv1d, bn: nn.BatchNorm1d | None) -> Tuple[np.ndarray, np.ndarray]:
    W = conv.weight.detach().cpu().numpy()[:, :, 0].astype(np.float64)
    b = (conv.bias.detach().cpu().numpy() if conv.bias is not None
         else np.zeros(W.shape[0])).astype(np.float64)
    if bn is not None:
        scale = bn.weight.detach().cpu().numpy() / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
        W = W * scale[:, None]
        b = (b - bn.running_mean.detach().cpu().numpy()) * scale + bn.bias.detach().cpu().numpy()
    return W.astype(np.float32), b.astype(np.float32)


def _block_layers(seq: nn.Sequential) -> List[Layer]:
    mods = list(seq)
    layers: List[Layer] = []
    i = 0
    while i < len(mods):
        conv = mods[i]
        assert isinstance(conv, nn.Conv1d), f"unexpected module {conv}"
        bn = mods[i + 1] if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm1d) else None
        W, b = _fold(conv, bn)
        i += 2 if bn is not None else 1
        relu = i < len(mods) and isinstance(mods[i], nn.ReLU)
        if relu:
            i += 1
        layers.append((W, b, relu))
    return layers


def folded_layers(model: PointNetRefiner) -> Dict[str, List[Layer]]:
    model = model.eval()
    head = _block_layers(model.head)
    head.append((*_fold(model.out, None), False))
    return {"local": _block_layers(model.local_mlp), "global": _block_layers(model.global_mlp), "head": head}


def export_weights(model: PointNetRefiner, path: str | Path, sectors: int = DEFAULT_SECTORS,
                   feature_scale: float = DEFAULT_FEATURE_SCALE, num_classes: int = NUM_CLASSES) -> None:
    blocks = folded_layers(model)
    cfg = model.cfg
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, cfg.in_features, num_classes, sectors))
        f.write(struct.pack("<fff", feature_scale, cfg.max_delta_range, cfg.max_delta_intensity))
        f.write(struct.pack("<III", len(blocks["local"]), len(blocks["global"]), len(blocks["head"])))
        for name in ("local", "global", "head"):
            for W, b, relu in blocks[name]:
                f.write(struct.pack("<III", W.shape[1], W.shape[0], int(relu)))
                f.write(np.ascontiguousarray(W, dtype="<f4").tobytes())
                f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())


def load_weights(path: str | Path) -> Dict:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("not a .ppnt file")
    off = 4
    version, in_f, n_cls, sectors = struct.unpack_from("<IIII", data, off); off += 16
    if version != VERSION:
        raise ValueError(f"unsupported .ppnt version {version}")
    scale, max_dr, max_di = struct.unpack_from("<fff", data, off); off += 12
    counts = struct.unpack_from("<III", data, off); off += 12
    blocks: Dict[str, List[Layer]] = {}
    for name, count in zip(("local", "global", "head"), counts):
        layers = []
        for _ in range(count):
            d_in, d_out, relu = struct.unpack_from("<III", data, off); off += 12
            W = np.frombuffer(data, "<f4", d_in * d_out, off).reshape(d_out, d_in); off += 4 * d_in * d_out
            b = np.frombuffer(data, "<f4", d_out, off); off += 4 * d_out
            layers.append((W, b, bool(relu)))
        blocks[name] = layers
    return {"in_features": in_f, "num_classes": n_cls, "sectors": sectors, "feature_scale": scale,
            "max_delta_range": max_dr, "max_delta_intensity": max_di, "blocks": blocks}


def reference_forward(weights: Dict, feats: np.ndarray) -> np.ndarray:
    """NumPy forward pass over one sector (N, F) -> raw outputs (N, 3); mirrors the C++ code."""
    def run(layers, x):
        for W, b, relu in layers:
            x = x @ W.T + b
            if relu:
                x = np.maximum(x, 0.0)
        return x

    blocks = weights["blocks"]
    local = run(blocks["local"], feats.astype(np.float32))
    glob = run(blocks["global"], local).max(axis=0, keepdims=True)
    x = np.concatenate([local, np.repeat(glob, len(local), axis=0)], axis=1)
    return run(blocks["head"], x)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a refiner checkpoint to .ppnt")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sectors", type=int, default=DEFAULT_SECTORS)
    ap.add_argument("--init-only", action="store_true",
                    help="export an untrained (identity) refiner, ignoring --checkpoint contents")
    args = ap.parse_args(argv)

    model = PointNetRefiner(RefinerConfig())
    if not args.init_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    export_weights(model, args.out, sectors=args.sectors)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
