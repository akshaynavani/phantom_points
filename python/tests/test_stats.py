import numpy as np

from phantom_points.stats import compare_samples, estimate_rings, histogram_pair, js_divergence


def test_js_divergence_bounds():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.5, 0.5])
    assert js_divergence(p, p) < 1e-9
    assert abs(js_divergence(p, q) - 1.0) < 1e-6
    assert js_divergence(p, q) == js_divergence(q, p)


def test_compare_samples_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    same = compare_samples(a, rng.normal(0, 1, 20000))
    shifted = compare_samples(a, rng.normal(1.0, 1, 20000))
    assert same["ks_stat"] < 0.03
    assert shifted["ks_stat"] > 0.3
    assert abs(shifted["wasserstein"] - 1.0) < 0.1
    assert shifted["js_divergence"] > same["js_divergence"]


def test_histogram_pair_shared_support():
    pa, pb, edges = histogram_pair(np.array([0.0, 1.0]), np.array([2.0, 3.0]), bins=3)
    assert edges[0] == 0.0 and edges[-1] == 3.0
    assert abs(pa.sum() - 1) < 1e-9 and abs(pb.sum() - 1) < 1e-9


def test_estimate_rings_nearest_elevation():
    elev = [-10.0, 0.0, 10.0]
    pts = []
    for e in (-9.0, 1.0, 12.0, -4.0):
        r = np.radians(e)
        pts.append([np.cos(r) * 10, 0.0, np.sin(r) * 10])
    rings = estimate_rings(np.array(pts), elev)
    assert rings.tolist() == [0, 1, 2, 1]
