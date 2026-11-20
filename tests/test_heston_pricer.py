import numpy as np
import pytest

from conftest import FO_HIGH_PRECISION_CALL, FO_PARAMS, FO_REFERENCE_CALL, random_cases
from heston_risk import HestonParams, bs_price, heston_price, heston_price_gil_pelaez, heston_price_quad, implied_vol


def test_fang_oosterlee_benchmark():
    for f in (heston_price, heston_price_quad, heston_price_gil_pelaez):
        px = f(100, 100, 1.0, 0.0, 0.0, FO_PARAMS)
        assert px == pytest.approx(FO_REFERENCE_CALL, abs=5e-8)       # published, 9 decimals
        assert px == pytest.approx(FO_HIGH_PRECISION_CALL, abs=1e-10)  # 30-digit mpmath


def test_gl_matches_independent_gil_pelaez_formula():
    errs = [abs(heston_price(100, K, T, 0.03, 0.01, p) - heston_price_gil_pelaez(100, K, T, 0.03, 0.01, p))
            for p, T, K in random_cases(60, seed=1)]
    assert max(errs) < 1e-8


def test_gl_matches_adaptive_quadrature():
    for p, T, K in random_cases(20, seed=2):
        assert heston_price(100, K, T, 0.02, 0.0, p) == pytest.approx(
            heston_price_quad(100, K, T, 0.02, 0.0, p), abs=1e-8)


def test_put_call_parity_against_directly_priced_puts(base_params):
    K = np.array([70.0, 100.0, 130.0])
    T, r, q = 1.5, 0.04, 0.015
    put_direct = heston_price_gil_pelaez(100, K, T, r, q, base_params, "put")
    put_lewis = heston_price(100, K, T, r, q, base_params, "put")
    call = heston_price(100, K, T, r, q, base_params, "call")
    assert np.allclose(put_lewis, put_direct, atol=1e-8)
    assert np.allclose(call - put_direct, 100 * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-8)


@pytest.mark.parametrize("rho", [0.0, -0.7])
def test_black_scholes_limit(rho):
    """sigma -> 0: variance is deterministic, price -> BS with the average
    expected variance. Error must shrink as sigma does (O(sigma) for rho != 0,
    O(sigma^2) for rho = 0)."""
    K = np.array([80.0, 100.0, 120.0])
    T = 0.75
    errs = []
    for s in [1e-1, 1e-2, 1e-3]:
        p = HestonParams(0.06, 1.5, 0.03, s, rho)
        bs = bs_price(100, K, T, 0.03, 0.01, np.sqrt(p.expected_avg_variance(T)))
        errs.append(np.max(np.abs(heston_price(100, K, T, 0.03, 0.01, p) - bs)))
    assert errs[0] > errs[1] > errs[2]
    order = np.log10(errs[1] / errs[2])  # observed convergence order in sigma
    if rho == 0:
        assert order == pytest.approx(2.0, abs=0.1) and errs[2] < 1e-5
    else:
        assert order == pytest.approx(1.0, abs=0.1) and errs[2] < 5e-3


def test_no_arbitrage_shape_in_strike(base_params):
    K = np.linspace(50, 200, 151)
    T, r, q = 0.5, 0.03, 0.01
    c = heston_price(100, K, T, r, q, base_params)
    F, df = 100 * np.exp((r - q) * T), np.exp(-r * T)
    assert np.all(c >= df * np.maximum(F - K, 0) - 1e-10)
    assert np.all(c <= 100 * np.exp(-q * T) + 1e-10)
    assert np.all(np.diff(c) < 0)                 # decreasing in K
    assert np.all(np.diff(c, 2) > -1e-10)         # convex in K


def test_negative_rho_gives_negative_skew(base_params):
    K = np.array([80.0, 100.0, 120.0])
    T = 1.0
    iv = implied_vol(heston_price(100, K, T, 0.0, 0.0, base_params), 100, K, T, 0.0, 0.0)
    assert iv[0] > iv[1] > iv[2]


def test_vectorised_multi_maturity_equals_loop(base_params):
    K = np.array([90.0, 100.0, 110.0, 95.0])
    T = np.array([0.25, 0.25, 1.0, 2.0])
    kind = np.array(["call", "put", "call", "put"])
    vec = heston_price(100, K, T, 0.03, 0.01, base_params, kind)
    loop = [heston_price(100, k, t, 0.03, 0.01, base_params, c) for k, t, c in zip(K, T, kind)]
    assert np.allclose(vec, loop, atol=1e-13)


def test_invalid_inputs(base_params):
    with pytest.raises(ValueError):
        heston_price(100, 100, 0.0, 0.0, 0.0, base_params)
    with pytest.raises(ValueError):
        heston_price(100, -1.0, 1.0, 0.0, 0.0, base_params)
