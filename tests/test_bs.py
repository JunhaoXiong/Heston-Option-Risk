import numpy as np
import pytest

from heston_risk.bs import bs_price, bs_vega, implied_vol


def test_textbook_value():
    # Hull: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert bs_price(100, 100, 1.0, 0.05, 0.0, 0.2) == pytest.approx(10.450583572185565, abs=1e-12)


def test_put_call_parity():
    K = np.linspace(50, 150, 21)
    c = bs_price(100, K, 0.7, 0.03, 0.01, 0.25, "call")
    p = bs_price(100, K, 0.7, 0.03, 0.01, 0.25, "put")
    assert np.allclose(c - p, 100 * np.exp(-0.01 * 0.7) - K * np.exp(-0.03 * 0.7), atol=1e-12)


def test_zero_vol_is_discounted_intrinsic():
    assert bs_price(100, 90, 1.0, 0.02, 0.0, 0.0) == pytest.approx(np.exp(-0.02) * (100 * np.exp(0.02) - 90))


def test_vega_matches_finite_difference():
    h = 1e-6
    fd = (bs_price(100, 110, 0.5, 0.03, 0.01, 0.2 + h) - bs_price(100, 110, 0.5, 0.03, 0.01, 0.2 - h)) / (2 * h)
    assert bs_vega(100, 110, 0.5, 0.03, 0.01, 0.2) == pytest.approx(fd, rel=1e-7)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    K = np.array([60, 80, 100, 120, 150.0])
    sig = np.array([0.45, 0.3, 0.2, 0.18, 0.25])
    px = bs_price(100, K, 1.2, 0.03, 0.01, sig, kind)
    assert np.allclose(implied_vol(px, 100, K, 1.2, 0.03, 0.01, kind), sig, atol=1e-9)


def test_implied_vol_nan_outside_arbitrage_bounds():
    intrinsic = np.exp(-0.03) * (100 * np.exp(0.03) - 80)
    assert np.isnan(implied_vol(intrinsic - 0.01, 100, 80, 1.0, 0.03, 0.0, "call"))
    assert np.isnan(implied_vol(100.5, 100, 80, 1.0, 0.03, 0.0, "call"))


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0, 0, 0.2, "straddle")
