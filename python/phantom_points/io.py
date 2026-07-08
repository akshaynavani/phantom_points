"""Readers for datasets written by ``phantom_gen``.

Layout of a generated dataset directory::

    dataset.json              sensor + generator metadata
    index.csv                 one row per scene
    velodyne/000000.bin       float32 [x, y, z, intensity] * N   (KITTI layout)
    labels/000000.label       uint32  semantic | (instance << 16)  (SemanticKITTI layout)
    aux/000000.aux            uint16  [ring, column, flags] * N
    boxes/000000.txt          one object per line (see BOX_COLUMNS)
    meta/000000.json          per-scene metadata
    ref/...                   optional paired scans from the reference sensor (same layout)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BOX_COLUMNS = [
    "class",
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw",
    "occlusion_level",
    "occlusion",
    "points",
    "speed",
    "instance",
]


@dataclass
class ScanData:
    xyz: np.ndarray        # (N, 3) float32
    intensity: np.ndarray  # (N,) float32
    label: np.ndarray      # (N,) uint16 semantic class
    instance: np.ndarray   # (N,) uint16
    ring: Optional[np.ndarray] = None    # (N,) uint16
    column: Optional[np.ndarray] = None  # (N,) uint16
    flags: Optional[np.ndarray] = None   # (N,) uint16

    @property
    def range(self) -> np.ndarray:
        return np.linalg.norm(self.xyz, axis=1)

    def __len__(self) -> int:
        return self.xyz.shape[0]


def scan_name(scene_id: int) -> str:
    return f"{scene_id:06d}"


def read_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_labels(path: Path):
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint16), (raw >> 16).astype(np.uint16)


def read_aux(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16).reshape(-1, 3)


def read_boxes(path: Path) -> List[Dict]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        row: Dict = {"class": parts[0]}
        for key, val in zip(BOX_COLUMNS[1:], parts[1:]):
            row[key] = float(val)
        for key in ("occlusion_level", "points", "instance"):
            row[key] = int(row[key])
        boxes.append(row)
    return boxes


class Dataset:
    """Random access to a generated dataset directory (optionally its ``ref/`` twin)."""

    def __init__(self, root: str | Path, reference: bool = False):
        self.root = Path(root)
        self.base = self.root / "ref" if reference else self.root
        meta_path = self.root / "dataset.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.ids = self._discover()

    def _discover(self) -> List[int]:
        index = self.root / "index.csv"
        if index.exists():
            with index.open() as f:
                return [int(r["scene_id"]) for r in csv.DictReader(f)]
        return sorted(int(p.stem) for p in (self.base / "velodyne").glob("*.bin"))

    @property
    def beams(self) -> int:
        return int(self.meta.get("sensor", {}).get("beams", 64))

    @property
    def columns(self) -> int:
        return int(self.meta.get("sensor", {}).get("columns", 2048))

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, scene_id: int) -> ScanData:
        name = scan_name(scene_id)
        pts = read_bin(self.base / "velodyne" / f"{name}.bin")
        label_path = self.base / "labels" / f"{name}.label"
        if label_path.exists():
            sem, inst = read_labels(label_path)
        else:
            sem = np.zeros(len(pts), np.uint16)
            inst = np.zeros(len(pts), np.uint16)
        scan = ScanData(xyz=pts[:, :3].copy(), intensity=pts[:, 3].copy(), label=sem, instance=inst)
        aux_path = self.base / "aux" / f"{name}.aux"
        if aux_path.exists():
            aux = read_aux(aux_path)
            scan.ring, scan.column, scan.flags = aux[:, 0], aux[:, 1], aux[:, 2]
        return scan

    def boxes(self, scene_id: int) -> List[Dict]:
        return read_boxes(self.root / "boxes" / f"{scan_name(scene_id)}.txt")

    def scene_meta(self, scene_id: int) -> Dict:
        p = self.root / "meta" / f"{scan_name(scene_id)}.json"
        return json.loads(p.read_text()) if p.exists() else {}
