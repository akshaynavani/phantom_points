import json

import numpy as np
import pytest

from phantom_points import FLAG_SPURIOUS
from phantom_points.features import build_features, num_features, sector_ids
from phantom_points.io import Dataset

COLUMNS = 16
BEAMS = 4


def write_scan(root, name, xyz, intensity, ring, column, flags=None, label=None):
    for sub in ("velodyne", "labels", "aux"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    np.hstack([xyz, intensity[:, None]]).astype(np.float32).tofile(root / "velodyne" / f"{name}.bin")
    lbl = np.zeros(n, np.uint32) if label is None else label.astype(np.uint32)
    lbl.tofile(root / "labels" / f"{name}.label")
    flags = np.zeros(n, np.uint16) if flags is None else flags
    np.stack([ring, column, flags], axis=1).astype(np.uint16).tofile(root / "aux" / f"{name}.aux")


@pytest.fixture
def paired(tmp_path):
    (tmp_path / "dataset.json").write_text(json.dumps({"sensor": {"beams": BEAMS, "columns": COLUMNS}}))
    (tmp_path / "index.csv").write_text("scene_id,layout,num_points,num_objects,seed\n0,straight,4,0,1\n")
    xyz = np.array([[10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]], np.float32)
    ring = np.array([0, 1, 2, 3])
    col = np.array([0, 4, 8, 12])
    write_scan(tmp_path, "000000", xyz, np.full(4, 0.2, np.float32), ring, col,
               label=np.array([1, 3, 3, 8]))
    # Reference: point 0 slightly farther, point 1 brighter, point 2 missing, point 3 spurious only.
    ref_xyz = np.array([[10.1, 0, 0], [0, 10, 0], [0, -10, 0]], np.float32)
    write_scan(tmp_path / "ref", "000000", ref_xyz, np.array([0.2, 0.3, 0.2], np.float32),
               np.array([0, 1, 3]), np.array([0, 4, 12]),
               flags=np.array([0, 0, FLAG_SPURIOUS], np.uint16))
    return tmp_path


def test_reader_round_trip(paired):
    ds = Dataset(paired)
    assert ds.ids == [0] and ds.beams == BEAMS and ds.columns == COLUMNS
    scan = ds.load(0)
    assert len(scan) == 4
    assert scan.label.tolist() == [1, 3, 3, 8]
    assert np.allclose(scan.range, 10.0)


def test_build_targets(paired):
    from phantom_points.dataset import build_targets

    src, ref = Dataset(paired).load(0), Dataset(paired, reference=True).load(0)
    t = build_targets(src, ref, COLUMNS)
    assert t["drop"].tolist() == [0, 0, 1, 1]  # spurious ref returns never count as matches
    assert t["reg_mask"].tolist() == [True, True, False, False]
    assert abs(t["d_range"][0] - 0.1) < 1e-5
    assert abs(t["d_int"][1] - 0.1) < 1e-6


def test_sector_dataset_shapes(paired):
    torch = pytest.importorskip("torch")
    from phantom_points.dataset import SectorPairDataset

    ds = SectorPairDataset(paired, sectors=4, max_points=8)
    assert len(ds) == 4
    item = ds[0]
    assert item["feats"].shape == (num_features(), 8)
    assert item["mask"].sum().item() == 1
    assert item["feats"].dtype == torch.float32


def test_feature_layout():
    xyz = np.array([[30.0, 40.0, 0.0]], np.float32)
    f = build_features(xyz, np.array([0.5]), np.array([63]), np.array([3]), beams=64, feature_scale=50.0)
    assert f.shape == (1, num_features())
    assert np.allclose(f[0, :6], [0.6, 0.8, 0.0, 1.0, 0.5, 1.0])
    assert f[0, 6 + 3] == 1.0 and f[0, 6:].sum() == 1.0
    assert sector_ids(np.array([0, 2047, 1024]), 2048, 32).tolist() == [0, 31, 16]
