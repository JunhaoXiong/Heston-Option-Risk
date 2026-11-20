import numpy as np
import pytest

from heston_risk import HestonParams, heston_price
from heston_risk.mc.euler import price_euler_mc


@pytest.mark.slow
@pytest.mark.parametrize("K,kind", [(80.0, "call"), (100.0, "call"), (120.0, "call"), (90.0, "put")])
def test_euler_mc_agrees_with_analytic(K, kind):
    """Independent sanity check: Feller-satisfying parameters, fine time grid.
    Allow 4 standard errors plus a small discretisation-bias allowance."""
    p = HestonParams(0.04, 2.0, 0.04, 0.3, -0.7)
    mc = price_euler_mc(100, K, 1.0, 0.03, 0.01, p, kind, n_paths=200_000, n_steps=250, seed=7)
    exact = heston_price(100, K, 1.0, 0.03, 0.01, p, kind)
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.01


def test_mc_is_reproducible():
    p = HestonParams(0.04, 2.0, 0.04, 0.3, -0.7)
    a = price_euler_mc(100, 100, 0.5, 0.0, 0.0, p, n_paths=2000, n_steps=20, seed=3)
    b = price_euler_mc(100, 100, 0.5, 0.0, 0.0, p, n_paths=2000, n_steps=20, seed=3)
    assert a == b
