import numpy as np
import pytest

from heston_risk import HestonParams


@pytest.mark.parametrize("kw", [dict(v0=-0.01), dict(kappa=0.0), dict(theta=-1.0), dict(sigma=0.0),
                                dict(rho=1.0), dict(rho=-1.0)])
def test_invalid_parameters_rejected(kw):
    base = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
    base.update(kw)
    with pytest.raises(ValueError):
        HestonParams(**base)


def test_feller():
    assert HestonParams(0.04, 2.0, 0.04, 0.3, -0.7).feller_satisfied
    assert not HestonParams(0.04, 2.0, 0.04, 0.5, -0.7).feller_satisfied  # 0.16/0.25 = 0.64


def test_expected_average_variance():
    p = HestonParams(0.09, 1.5, 0.04, 0.5, -0.5)
    T = 2.0
    t = np.linspace(0, T, 200001)
    v = p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * t)  # E[v_t]
    assert p.expected_avg_variance(T) == pytest.approx(np.trapezoid(v, t) / T, rel=1e-9)
