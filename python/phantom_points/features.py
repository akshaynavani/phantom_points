"""Per-point feature construction and sector partitioning.

This module is the single source of truth for the refiner's input layout. The C++
implementation in ``src/pointnet.cpp`` (``PointNetRefiner::build_features``) must
produce exactly the same vector for each point:

    [x/S, y/S, z/S, range/S, intensity, ring/(beams-1), one_hot(label, C)]

where ``S`` is ``feature_scale`` and ``C`` is the number of semantic classes.
"""

from __future__ import annotations

import numpy as np

from . import NUM_CLASSES

NUM_GEOMETRIC_FEATURES = 6
DEFAULT_FEATURE_SCALE = 50.0
DEFAULT_SECTORS = 32


def num_features(num_classes: int = NUM_CLASSES) -> int:
    return NUM_GEOMETRIC_FEATURES + num_classes


def build_features(
    xyz: np.ndarray,
    intensity: np.ndarray,
    ring: np.ndarray,
    label: np.ndarray,
    beams: int,
    feature_scale: float = DEFAULT_FEATURE_SCALE,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Return an ``(N, 6 + C)`` float32 feature matrix."""
    n = xyz.shape[0]
    rng = np.linalg.norm(xyz, axis=1)
    feats = np.zeros((n, num_features(num_classes)), dtype=np.float32)
    feats[:, 0:3] = xyz / feature_scale
    feats[:, 3] = rng / feature_scale
    feats[:, 4] = intensity
    feats[:, 5] = ring / max(beams - 1, 1)
    lbl = np.clip(label.astype(np.int64), 0, num_classes - 1)
    feats[np.arange(n), NUM_GEOMETRIC_FEATURES + lbl] = 1.0
    return feats


def sector_ids(column: np.ndarray, columns: int, sectors: int = DEFAULT_SECTORS) -> np.ndarray:
    """Azimuth sector index for each point (matches the C++ integer arithmetic)."""
    return (column.astype(np.int64) * sectors) // columns
