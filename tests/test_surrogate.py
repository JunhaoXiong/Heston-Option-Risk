import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heston_risk import HestonParams, heston_price, implied_vol  # noqa: E402
from heston_risk.surrogate.dataset import BOX_HI, BOX_LO, K_WIDTH, T_MAX, T_MIN, label_slice, make_dataset, sample_slices  # noqa: E402
from heston_risk.surrogate.model import IVNet, Surrogate, in_domain, scale_inputs  # noqa: E402
from heston_risk.surrogate.train import TrainConfig, train_surrogate  # noqa: E402


def test_slices_inside_box_and_maturity_range():
    s = sample_slices(1024, seed=0)
    assert np.all(s[:, :5] >= BOX_LO) and np.all(s[:, :5] <= BOX_HI)
    assert s[:, 5].min() >= T_MIN and s[:, 5].max() <= T_MAX


def test_labels_match_analytic_pricer_at_any_spot_and_rate():
    """IV depends only on (params, k, T): a label computed at F = 1, r = 0 must equal the IV of the
    same option priced at S = 693, r = 4%, q = 1%."""
    row = np.array([0.05, 2.0, 0.04, 0.8, -0.7, 0.5])
    ks = np.array([-0.2, 0.0, 0.15])
    iv, _ = label_slice(row, ks)
    S, r, q, T = 693.0, 0.04, 0.01, 0.5
    F = S * np.exp((r - q) * T)
    K = F * np.exp(ks)
    kind = np.where(ks < 0, "put", "call")
    px = heston_price(S, K, T, r, q, HestonParams(*row[:5]), kind)
    assert np.allclose(implied_vol(px, S, K, T, r, q, kind), iv, atol=1e-9)


def test_dataset_rows_keep_slice_structure():
    d = make_dataset(64, 5, seed=3, workers=2)
    assert d["X"].shape[1] == 7 and len(d["y"]) == len(d["slice_id"])
    assert d["n_total"] == 64 * 5 and d["n_total"] - d["n_dropped"] == len(d["y"])
    # all rows of a slice share parameters and maturity
    for s in np.unique(d["slice_id"])[:10]:
        rows = d["X"][d["slice_id"] == s]
        assert np.all(rows[:, [0, 1, 2, 3, 4, 6]] == rows[0, [0, 1, 2, 3, 4, 6]])
    assert np.all(np.abs(d["X"][:, 5]) <= K_WIDTH * np.sqrt(d["X"][:, 6]) + 1e-12)


def test_input_scaling_maps_box_to_unit_interval():
    X = np.array([[*BOX_LO, -K_WIDTH * np.sqrt(T_MIN), T_MIN], [*BOX_HI, K_WIDTH * np.sqrt(T_MAX), T_MAX]])
    S = scale_inputs(X)
    assert np.allclose(S[0], -1) and np.allclose(S[1], 1)
    assert in_domain(X).all()
    X_out = X.copy()
    X_out[0, 3] = 4.0  # sigma above the box
    assert not in_domain(X_out)[0]


def test_tiny_training_run_learns_and_save_load_round_trip(tmp_path):
    d = make_dataset(256, 8, seed=11, workers=2)
    tr = d["slice_id"] < 200
    sur, log = train_surrogate(d["X"][tr], d["y"][tr], d["X"][~tr], d["y"][~tr],
                               TrainConfig(width=32, depth=2, max_epochs=15, batch_size=256), verbose=False)
    assert log.val_rmse[-1] < log.val_rmse[0]          # it learns something
    path = tmp_path / "m.pt"
    sur.save(path)
    sur2 = Surrogate.load(path)
    assert np.allclose(sur.iv(d["X"][:20]), sur2.iv(d["X"][:20]))


def test_param_jacobian_matches_finite_differences():
    torch.manual_seed(0)
    sur = Surrogate(IVNet(16, 2))
    X = np.array([[0.05, 2.0, 0.04, 0.8, -0.6, 0.05, 0.5], [0.1, 5.0, 0.08, 1.5, -0.3, -0.1, 1.2]])
    _, jac = sur.iv_and_param_jacobian(X)
    h = 1e-3
    for i in range(5):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, i] += h
        Xm[:, i] -= h
        fd = (sur.iv(Xp) - sur.iv(Xm)) / (2 * h)
        assert np.allclose(jac[:, i], fd, atol=2e-5)
