"""Paired training data for the refiner.

A training pair is the *same scene* rendered by two sensors that share the beam layout:
the synthetic source (``<root>/velodyne``) and a reference (``<root>/ref/velodyne``),
which can be a higher-fidelity simulator profile or real scans re-projected onto the
same (ring, column) grid. Points are matched by their (ring, column) cell:

* matched      -> regress ``range_ref - range_src`` and ``intensity_ref - intensity_src``
* unmatched    -> drop target = 1 (the reference sensor saw nothing in that cell)

Each item is one azimuth sector of one scan, subsampled / padded to ``max_points``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from . import FLAG_SPURIOUS
from .features import DEFAULT_FEATURE_SCALE, DEFAULT_SECTORS, build_features, sector_ids
from .io import Dataset, ScanData


def _cell_keys(scan: ScanData, columns: int) -> np.ndarray:
    return scan.ring.astype(np.int64) * columns + scan.column.astype(np.int64)


def build_targets(src: ScanData, ref: ScanData, columns: int) -> Dict[str, np.ndarray]:
    """Per-source-point regression / dropout targets from (ring, column) matching."""
    ref_ok = (ref.flags & FLAG_SPURIOUS) == 0 if ref.flags is not None else np.ones(len(ref), bool)
    ref_keys = _cell_keys(ref, columns)[ref_ok]
    ref_range = ref.range[ref_ok]
    ref_int = ref.intensity[ref_ok]
    order = np.argsort(ref_keys, kind="stable")
    ref_keys, ref_range, ref_int = ref_keys[order], ref_range[order], ref_int[order]

    src_keys = _cell_keys(src, columns)
    if len(ref_keys) == 0:
        # Reference saw nothing at all: every source point should be dropped.
        ref_keys = np.array([-1], np.int64)
        ref_range = np.zeros(1, np.float32)
        ref_int = np.zeros(1, np.float32)
    pos = np.searchsorted(ref_keys, src_keys)
    pos_c = np.clip(pos, 0, len(ref_keys) - 1)
    matched = (pos < len(ref_keys)) & (ref_keys[pos_c] == src_keys)
    src_spurious = (src.flags & FLAG_SPURIOUS) != 0 if src.flags is not None else np.zeros(len(src), bool)

    d_range = np.where(matched, ref_range[pos_c] - src.range, 0.0).astype(np.float32)
    d_int = np.where(matched, ref_int[pos_c] - src.intensity, 0.0).astype(np.float32)
    # Large range jumps mean the cells hit different surfaces; don't regress on them.
    reg_mask = matched & (np.abs(d_range) < 1.0) & ~src_spurious
    drop = (~matched).astype(np.float32)
    return {"d_range": d_range, "d_int": d_int, "reg_mask": reg_mask, "drop": drop}


class SectorPairDataset(TorchDataset):
    def __init__(
        self,
        root: str | Path,
        scene_ids: List[int] | None = None,
        sectors: int = DEFAULT_SECTORS,
        max_points: int = 4096,
        feature_scale: float = DEFAULT_FEATURE_SCALE,
        seed: int = 0,
    ):
        self.src = Dataset(root)
        self.ref = Dataset(root, reference=True)
        self.ids = scene_ids if scene_ids is not None else list(self.src.ids)
        self.sectors = sectors
        self.max_points = max_points
        self.feature_scale = feature_scale
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.ids) * self.sectors

    def _scene(self, scene_id: int):
        if scene_id not in self._cache:
            src = self.src.load(scene_id)
            ref = self.ref.load(scene_id)
            if src.ring is None or ref.ring is None:
                raise ValueError(f"scene {scene_id}: aux files are required for pairing")
            feats = build_features(src.xyz, src.intensity, src.ring, src.label,
                                   self.src.beams, self.feature_scale)
            targets = build_targets(src, ref, self.src.columns)
            sec = sector_ids(src.column, self.src.columns, self.sectors)
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[scene_id] = (feats, targets, sec)
        return self._cache[scene_id]

    def __getitem__(self, idx: int):
        scene_id = self.ids[idx // self.sectors]
        sector = idx % self.sectors
        feats, targets, sec = self._scene(scene_id)
        sel = np.flatnonzero(sec == sector)
        if len(sel) > self.max_points:
            sel = self.rng.choice(sel, self.max_points, replace=False)

        n, F = len(sel), feats.shape[1]
        x = np.zeros((F, self.max_points), np.float32)
        mask = np.zeros(self.max_points, bool)
        x[:, :n] = feats[sel].T
        mask[:n] = True
        out = {"feats": torch.from_numpy(x), "mask": torch.from_numpy(mask)}
        for key in ("d_range", "d_int", "drop"):
            t = np.zeros(self.max_points, np.float32)
            t[:n] = targets[key][sel]
            out[key] = torch.from_numpy(t)
        reg = np.zeros(self.max_points, bool)
        reg[:n] = targets["reg_mask"][sel]
        out["reg_mask"] = torch.from_numpy(reg)
        return out
