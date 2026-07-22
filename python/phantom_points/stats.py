"""Statistical distribution analysis for synthetic vs reference LiDAR datasets.

Per-point features are pooled over a sample of scans and compared with

* two-sample Kolmogorov–Smirnov statistic (shape difference, scale free),
* 1-D Wasserstein / earth mover's distance (in the feature's units),
* Jensen–Shannon divergence on a shared histogram (bounded in [0, 1], base 2).

Scan-level features (points per scan, return ratio, points per ring) and label-level
summaries (class mix, occlusion levels, object point counts vs. distance) complete the
report. Everything returns plain dicts so reports serialise straight to JSON.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats as sps

from . import CLASS_NAMES, FLAG_MIXED_PIXEL, FLAG_SPURIOUS
from .io import Dataset, ScanData

POINT_FEATURES = ("range", "intensity", "z", "elevation_deg")
DEFAULT_BINS = 100


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------

def histogram_pair(a: np.ndarray, b: np.ndarray, bins: int = DEFAULT_BINS,
                   value_range: Optional[Sequence[float]] = None):
    """Normalised histograms of ``a`` and ``b`` over a shared support."""
    if value_range is None:
        lo = float(min(np.min(a), np.min(b)))
        hi = float(max(np.max(a), np.max(b)))
        if hi <= lo:
            hi = lo + 1e-6
        value_range = (lo, hi)
    ha, edges = np.histogram(a, bins=bins, range=value_range)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return pa, pb, edges


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (base 2, in [0, 1]) between two discrete distributions."""
    p = np.asarray(p, np.float64) + eps
    q = np.asarray(q, np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float(np.sum(x * np.log2(x / y)))  # noqa: E731
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, np.float64) + eps
    q = np.asarray(q, np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compare_samples(a: np.ndarray, b: np.ndarray, bins: int = DEFAULT_BINS,
                    max_samples: int = 200_000, seed: int = 0) -> Dict[str, float]:
    """Compare two 1-D samples; large inputs are subsampled for the KS / EMD tests."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"n_a": len(a), "n_b": len(b)}
    rng = np.random.default_rng(seed)
    sa = rng.choice(a, max_samples, replace=False) if len(a) > max_samples else a
    sb = rng.choice(b, max_samples, replace=False) if len(b) > max_samples else b
    ks = sps.ks_2samp(sa, sb)
    pa, pb, _ = histogram_pair(a, b, bins)
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std()),
        "std_b": float(b.std()),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(sps.wasserstein_distance(sa, sb)),
        "js_divergence": js_divergence(pa, pb),
        "kl_divergence": kl_divergence(pa, pb),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def estimate_rings(xyz: np.ndarray, elevations_deg: Sequence[float]) -> np.ndarray:
    """Assign each point to the nearest beam elevation (for scans without ring metadata)."""
    elev = np.degrees(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)))
    table = np.asarray(sorted(elevations_deg), np.float64)
    idx = np.clip(np.searchsorted(table, elev), 1, len(table) - 1)
    left, right = table[idx - 1], table[idx]
    return np.where(np.abs(elev - left) <= np.abs(elev - right), idx - 1, idx).astype(np.int64)


def point_features(scan: ScanData, exclude_spurious: bool = False) -> Dict[str, np.ndarray]:
    keep = np.ones(len(scan), bool)
    if exclude_spurious and scan.flags is not None:
        keep &= (scan.flags & FLAG_SPURIOUS) == 0
    xyz = scan.xyz[keep]
    r = np.linalg.norm(xyz, axis=1)
    return {
        "range": r,
        "intensity": scan.intensity[keep],
        "z": xyz[:, 2],
        "elevation_deg": np.degrees(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))),
    }


def ring_histogram(scan: ScanData, beams: int, elevations_deg: Optional[Sequence[float]] = None) -> np.ndarray:
    ring = scan.ring.astype(np.int64) if scan.ring is not None else (
        estimate_rings(scan.xyz, elevations_deg) if elevations_deg is not None else None)
    if ring is None:
        return np.zeros(beams)
    return np.bincount(np.clip(ring, 0, beams - 1), minlength=beams).astype(np.float64)


class FeaturePool:
    """Accumulates features over many scans of one dataset."""

    def __init__(self, beams: int, columns: int, elevations_deg: Optional[Sequence[float]] = None):
        self.beams = beams
        self.columns = columns
        self.elevations_deg = elevations_deg
        self.points: Dict[str, List[np.ndarray]] = {k: [] for k in POINT_FEATURES}
        self.points_per_scan: List[int] = []
        self.ring_counts = np.zeros(beams)
        self.class_counts = np.zeros(len(CLASS_NAMES))
        self.intensity_by_class: Dict[int, List[np.ndarray]] = {}
        self.mixed = 0
        self.spurious = 0
        self.total = 0

    def add(self, scan: ScanData) -> None:
        feats = point_features(scan)
        for k in POINT_FEATURES:
            self.points[k].append(feats[k].astype(np.float32))
        self.points_per_scan.append(len(scan))
        self.ring_counts += ring_histogram(scan, self.beams, self.elevations_deg)
        self.total += len(scan)
        if scan.label is not None and len(scan.label):
            lbl = np.clip(scan.label.astype(np.int64), 0, len(CLASS_NAMES) - 1)
            self.class_counts += np.bincount(lbl, minlength=len(CLASS_NAMES))
            for c in np.unique(lbl):
                self.intensity_by_class.setdefault(int(c), []).append(scan.intensity[lbl == c])
        if scan.flags is not None:
            self.mixed += int(np.count_nonzero(scan.flags & FLAG_MIXED_PIXEL))
            self.spurious += int(np.count_nonzero(scan.flags & FLAG_SPURIOUS))

    def feature(self, name: str) -> np.ndarray:
        parts = self.points[name]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def summary(self) -> Dict:
        pps = np.asarray(self.points_per_scan, np.float64)
        out: Dict = {
            "scans": len(pps),
            "points_per_scan": describe(pps),
            "return_ratio": float(pps.mean() / (self.beams * self.columns)) if len(pps) else 0.0,
            "mixed_pixel_fraction": self.mixed / max(self.total, 1),
            "spurious_fraction": self.spurious / max(self.total, 1),
            "features": {k: describe(self.feature(k)) for k in POINT_FEATURES},
        }
        if self.class_counts.sum() > 0:
            frac = self.class_counts / self.class_counts.sum()
            out["class_fraction"] = {CLASS_NAMES[i]: float(f) for i, f in enumerate(frac)}
            out["intensity_by_class"] = {
                CLASS_NAMES[c]: describe(np.concatenate(v)) for c, v in sorted(self.intensity_by_class.items())
            }
        return out


def describe(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "max": float(x.max()),
        "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "p50": float(q[3]),
        "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
    }


def pool_dataset(ds: Dataset, scene_ids: Iterable[int], elevations_deg=None) -> FeaturePool:
    elevations = elevations_deg or ds.meta.get("sensor", {}).get("elevations_deg")
    pool = FeaturePool(ds.beams, ds.columns, elevations)
    for sid in scene_ids:
        pool.add(ds.load(sid))
    return pool


# ---------------------------------------------------------------------------
# Label-level statistics
# ---------------------------------------------------------------------------

def object_statistics(ds: Dataset, scene_ids: Iterable[int]) -> Dict:
    """Class counts, occlusion level mix and points-per-object vs. distance."""
    per_class = Counter()
    occ_levels = Counter()
    dist_bins = np.array([0, 10, 20, 30, 40, 50, 60, 80, 120])
    pts_by_bin: Dict[str, List[List[int]]] = {}
    layouts = Counter()
    for sid in scene_ids:
        layouts[ds.scene_meta(sid).get("layout", "unknown")] += 1
        for b in ds.boxes(sid):
            cls = b["class"]
            per_class[cls] += 1
            occ_levels[int(b["occlusion_level"])] += 1
            d = float(np.hypot(b["x"], b["y"]))
            k = int(np.clip(np.searchsorted(dist_bins, d) - 1, 0, len(dist_bins) - 2))
            pts_by_bin.setdefault(cls, [[] for _ in range(len(dist_bins) - 1)])[k].append(b["points"])
    return {
        "layouts": dict(layouts),
        "objects_per_class": dict(per_class),
        "occlusion_levels": {str(k): v for k, v in sorted(occ_levels.items())},
        "distance_bins_m": dist_bins.tolist(),
        "median_points_by_distance": {
            cls: [float(np.median(v)) if v else None for v in bins] for cls, bins in pts_by_bin.items()
        },
    }


# ---------------------------------------------------------------------------
# Dataset comparison
# ---------------------------------------------------------------------------

def compare_pools(a: FeaturePool, b: FeaturePool, bins: int = DEFAULT_BINS) -> Dict:
    report: Dict = {"point_features": {}}
    for k in POINT_FEATURES:
        report["point_features"][k] = compare_samples(a.feature(k), b.feature(k), bins)
    report["points_per_scan"] = compare_samples(np.asarray(a.points_per_scan), np.asarray(b.points_per_scan), 30)
    if a.beams == b.beams and a.ring_counts.sum() > 0 and b.ring_counts.sum() > 0:
        pa = a.ring_counts / a.ring_counts.sum()
        pb = b.ring_counts / b.ring_counts.sum()
        report["ring_distribution_js"] = js_divergence(pa, pb)
    if a.class_counts.sum() > 0 and b.class_counts.sum() > 0:
        report["class_distribution_js"] = js_divergence(a.class_counts, b.class_counts)
    # Single headline number: mean JS divergence over point features (lower is better).
    js = [v.get("js_divergence") for v in report["point_features"].values() if "js_divergence" in v]
    report["realism_gap"] = float(np.mean(js)) if js else None
    return report
