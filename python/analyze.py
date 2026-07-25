#!/usr/bin/env python3
"""Statistical report for a generated dataset, optionally against a reference set.

    # describe a synthetic dataset
    python python/analyze.py --synthetic out/dataset --out reports/base

    # compare against its paired reference-sensor scans
    python python/analyze.py --synthetic out/pairs --reference out/pairs --reference-ref

    # compare against real scans (any directory with velodyne/*.bin in KITTI format)
    python python/analyze.py --synthetic out/dataset --reference /data/kitti/sequences/00 --plots
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phantom_points.io import Dataset  # noqa: E402
from phantom_points.stats import (  # noqa: E402
    POINT_FEATURES,
    compare_pools,
    object_statistics,
    pool_dataset,
)


def sample_ids(ds: Dataset, n: int, seed: int):
    ids = list(ds.ids)
    if n and len(ids) > n:
        ids = random.Random(seed).sample(ids, n)
    return sorted(ids)


def make_plots(out: Path, syn, ref=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, POINT_FEATURES):
        a = syn.feature(name)
        kw = dict(bins=100, density=True, histtype="step", linewidth=1.4)
        ax.hist(a, label="synthetic", **kw)
        if ref is not None:
            ax.hist(ref.feature(name), label="reference", **kw)
        ax.set_title(name)
        if name in ("range", "intensity"):
            ax.set_yscale("log")
        ax.legend()

    ax = axes.flat[4]
    ax.plot(syn.ring_counts / max(syn.ring_counts.sum(), 1), label="synthetic")
    if ref is not None and ref.ring_counts.sum() > 0:
        ax.plot(ref.ring_counts / ref.ring_counts.sum(), label="reference")
    ax.set_title("points per ring (normalised)")
    ax.set_xlabel("ring")
    ax.legend()

    ax = axes.flat[5]
    ax.hist(np.asarray(syn.points_per_scan), bins=30, alpha=0.6, label="synthetic")
    if ref is not None:
        ax.hist(np.asarray(ref.points_per_scan), bins=30, alpha=0.6, label="reference")
    ax.set_title("points per scan")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", required=True, help="dataset produced by phantom_gen")
    ap.add_argument("--reference", help="reference dataset (phantom_gen output or KITTI-style dir)")
    ap.add_argument("--reference-ref", action="store_true",
                    help="use the paired ref/ scans inside --reference")
    ap.add_argument("--max-scans", type=int, default=200, help="scans sampled per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/latest")
    ap.add_argument("--plots", action="store_true", help="write distributions.png (needs matplotlib)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    syn_ds = Dataset(args.synthetic)
    syn_ids = sample_ids(syn_ds, args.max_scans, args.seed)
    print(f"synthetic: {len(syn_ds)} scans, analysing {len(syn_ids)}")
    syn = pool_dataset(syn_ds, syn_ids)
    report = {
        "synthetic": {"path": args.synthetic, **syn.summary(), "objects": object_statistics(syn_ds, syn_ids)},
    }

    ref = None
    if args.reference:
        ref_ds = Dataset(args.reference, reference=args.reference_ref)
        ref_ids = sample_ids(ref_ds, args.max_scans, args.seed + 1)
        print(f"reference: {len(ref_ds)} scans, analysing {len(ref_ids)}")
        # Real scans lack ring metadata; estimate rings from the synthetic beam table.
        ref = pool_dataset(ref_ds, ref_ids, elevations_deg=syn.elevations_deg)
        report["reference"] = {"path": args.reference, **ref.summary()}
        report["comparison"] = compare_pools(syn, ref)

        print("\nfeature          KS     EMD      JS")
        for name, row in report["comparison"]["point_features"].items():
            if "ks_stat" in row:
                print(f"  {name:<14} {row['ks_stat']:.3f}  {row['wasserstein']:7.3f}  {row['js_divergence']:.4f}")
        gap = report["comparison"]["realism_gap"]
        if gap is not None:
            print(f"\nrealism gap (mean JS over point features): {gap:.4f}")

    (out / "report.json").write_text(json.dumps(report, indent=2))
    if args.plots:
        make_plots(out, syn, ref)
    print(f"\nwrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
