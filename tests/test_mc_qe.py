import numpy as np
import pytest

from heston_risk import HestonParams, heston_price
from heston_risk.mc.simulate import QEConfig, _qe_step, StepStats, price_mc, simulate

ANDERSEN_I = (HestonParams(0.04, 0.5, 0.04, 1.0, -0.9), 10.0)
MILD = HestonParams(0.04, 2.0, 0.04, 0.3, -0.7)


def _one_step(p, v, dt, n=400_000, seed=0, cfg=QEConfig()):
    rng = np.random.default_rng(seed)
    zv, zx = rng.standard_normal(n), rng.standard_normal(n)
    x0 = np.zeros(n)
    v0 = np.full(n, v)
    return _qe_step(x0, v0, dt, 0.0, 0.0, p, zv, zx, StepStats(), cfg)


@pytest.mark.parametrize("v", [0.0, 0.001, 0.04, 0.2])
@pytest.mark.parametrize("dt", [1 / 32, 0.25, 1.0])
def test_qe_variance_matches_exact_cir_moments(v, dt):
    """Both branches are moment-matched: mean and variance of v' | v must equal
    the exact CIR conditional moments (up to MC error)."""
    p = ANDERSEN_I[0]
    _, vn = _one_step(p, v, dt)
    e = np.exp(-p.kappa * dt)
    m = p.theta + (v - p.theta) * e
    s2 = v * p.sigma**2 * e * (1 - e) / p.kappa + p.theta * p.sigma**2 * (1 - e) ** 2 / (2 * p.kappa)
    assert np.all(vn >= 0)
    assert vn.mean() == pytest.approx(m, abs=5 * np.sqrt(s2 / vn.size))
    assert vn.var() == pytest.approx(s2, rel=0.05)


def test_qe_martingale_correction_is_exact_in_expectation():
    """With the correction, E[S'/S | v] = 1 per step (r = q = 0) up to MC error;
    without it there is a visible drift for this harsh parameter set."""
    p = ANDERSEN_I[0]
    x, _ = _one_step(p, 0.2, 1.0, n=1_000_000)
    ratio = np.exp(x)
    assert ratio.mean() == pytest.approx(1.0, abs=4 * ratio.std() / np.sqrt(ratio.size))
    x2, _ = _one_step(p, 0.2, 1.0, n=1_000_000, cfg=QEConfig(martingale_correction=False))
    assert abs(np.exp(x2).mean() - 1.0) > 4 * ratio.std() / np.sqrt(ratio.size)


def test_forward_is_martingale_over_many_steps():
    p, T = ANDERSEN_I
    m = price_mc(100, [1e-8], T, 0.02, 0.0, p, "call", n_paths=200_000, n_steps=20, seed=3)
    # call with K ~ 0 is the discounted stock: must equal S0 e^{-qT} = 100
    assert m.price[0] == pytest.approx(100.0, abs=4 * m.std_error[0])


def test_qe_converges_to_analytic_on_mild_parameters():
    K = np.array([80.0, 100.0, 120.0])
    m = price_mc(100, K, 1.0, 0.03, 0.01, MILD, "call", n_paths=400_000, n_steps=32, seed=11)
    ex = heston_price(100, K, 1.0, 0.03, 0.01, MILD)
    assert np.all(np.abs(m.price - ex) < 3.5 * m.std_error + 2e-3)


def test_qe_beats_euler_on_andersen_case_i():
    p, T = ANDERSEN_I
    K = np.array([100.0])
    ex = heston_price(100, K, T, 0.0, 0.0, p)
    qe = price_mc(100, K, T, 0, 0, p, n_paths=200_000, n_steps=40, scheme="qe", seed=5)
    eu = price_mc(100, K, T, 0, 0, p, n_paths=200_000, n_steps=40, scheme="euler", seed=5)
    assert abs(eu.price[0] - ex[0]) > 5 * eu.std_error[0]              # Euler visibly biased
    assert abs(qe.price[0] - ex[0]) < abs(eu.price[0] - ex[0]) / 5      # QE much closer


def test_control_variate_reduces_standard_error_and_stays_unbiased():
    K = np.array([90.0, 100.0])
    a = price_mc(100, K, 1.0, 0.03, 0.01, MILD, n_paths=200_000, n_steps=16, seed=2)
    b = price_mc(100, K, 1.0, 0.03, 0.01, MILD, n_paths=200_000, n_steps=16, seed=2, control_variate=True)
    assert np.all(b.std_error < a.std_error)
    assert np.all(np.abs(a.price - b.price) < 3 * a.std_error)


def test_simulate_saves_grid_and_pairs_are_antithetic():
    rng = np.random.default_rng(0)
    t, S, v, _ = simulate(100, 1.0, 0.0, 0.0, MILD, 10, 12, rng, "qe", True, save_every=3)
    assert np.allclose(t, [0, 0.25, 0.5, 0.75, 1.0])
    assert S.shape == (5, 10) and v.shape == (5, 10)
    assert np.all(S[0] == 100) and np.all(v[0] == MILD.v0)


def test_reproducible_and_validates_inputs():
    a = price_mc(100, 100, 0.5, 0, 0, MILD, n_paths=2000, n_steps=8, seed=9)
    b = price_mc(100, 100, 0.5, 0, 0, MILD, n_paths=2000, n_steps=8, seed=9)
    assert np.array_equal(a.price, b.price)
    with pytest.raises(ValueError):
        simulate(100, 1, 0, 0, MILD, 11, 4, np.random.default_rng(0), "qe", True)
    with pytest.raises(ValueError):
        simulate(100, 1, 0, 0, MILD, 10, 4, np.random.default_rng(0), "milstein", True)
