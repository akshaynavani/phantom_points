import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom_points.export import export_weights, load_weights, reference_forward  # noqa: E402
from phantom_points.features import num_features  # noqa: E402
from phantom_points.model import PointNetRefiner  # noqa: E402


def _randomise(model):
    torch.manual_seed(0)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    torch.nn.init.normal_(model.out.weight, std=0.1)
    torch.nn.init.normal_(model.out.bias, std=0.1)


def test_forward_shapes_and_mask():
    model = PointNetRefiner().eval()
    x = torch.randn(2, num_features(), 50)
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[1, 25:] = False
    out = model(x, mask)
    assert out.shape == (2, 3, 50)
    d_r, d_i, logit = model.decode(out)
    assert d_r.abs().max() <= model.cfg.max_delta_range


def test_new_model_is_identity():
    model = PointNetRefiner().eval()
    out = model(torch.randn(1, num_features(), 10))
    assert torch.allclose(out, torch.zeros_like(out))


def test_export_matches_torch(tmp_path):
    model = PointNetRefiner()
    _randomise(model)
    model.eval()
    path = tmp_path / "m.ppnt"
    export_weights(model, path, sectors=8)
    w = load_weights(path)
    assert w["sectors"] == 8 and w["in_features"] == num_features()

    feats = np.random.default_rng(1).normal(size=(64, num_features())).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(feats.T[None]))[0].numpy().T
    got = reference_forward(w, feats)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
