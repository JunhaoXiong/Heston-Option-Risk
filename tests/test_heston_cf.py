import numpy as np
import pytest

from heston_risk import HestonParams
from heston_risk.heston.charfunc import cf_heston1993, cf_little_trap


def test_normalisation_and_martingale(base_params):
    for T in [0.05, 1.0, 10.0]:
        assert cf_little_trap(0.0, T, base_params) == pytest.approx(1.0, abs=1e-14)
        # E[S_T / F_T] = 1  <=>  phi(-i) = 1
        assert cf_little_trap(-1j, T, base_params) == pytest.approx(1.0, abs=1e-12)


def test_hermitian_symmetry(base_params):
    u = np.linspace(0.1, 50, 100)
    assert np.allclose(cf_little_trap(-u, 1.0, base_params), np.conj(cf_little_trap(u, 1.0, base_params)))


def test_modulus_bounded_on_real_axis(base_params):
    u = np.linspace(0, 200, 2001)
    assert np.all(np.abs(cf_little_trap(u, 2.0, base_params)) <= 1.0 + 1e-12)


def test_gaussian_limit_cumulants():
    # sigma -> 0, rho = 0: X_T ~ N(-w/2, w) with w = integrated expected variance
    p = HestonParams(0.06, 1.5, 0.03, 1e-4, 0.0)
    T = 0.75
    w = p.expected_avg_variance(T) * T
    u = np.linspace(0, 20, 50)
    assert np.allclose(cf_little_trap(u, T, p), np.exp(-0.5j * u * w - 0.5 * u**2 * w), atol=1e-7)


def test_forms_agree_when_no_branch_crossing():
    p = HestonParams(0.04, 2.0, 0.04, 0.3, -0.5)
    u = np.linspace(0, 10, 200) - 0.5j
    assert np.allclose(cf_heston1993(u, 0.25, p), cf_little_trap(u, 0.25, p), atol=1e-12)


def test_original_form_jumps_branch_for_long_maturity():
    """Documented failure mode: with the principal complex log, the original
    Heston (1993) CF becomes discontinuous along u for T = 5y, while the
    little-trap form stays continuous."""
    p = HestonParams(0.04, 2.0, 0.04, 0.6, -0.8)
    u = np.linspace(0, 20, 20001) - 0.5j
    with np.errstate(all="ignore"):
        bad = cf_heston1993(u, 5.0, p)
    good = cf_little_trap(u, 5.0, p)
    assert np.max(np.abs(bad - good)) > 0.1
    assert np.max(np.abs(np.diff(good))) < 1e-3          # continuous
    assert np.max(np.abs(np.diff(bad))) > 0.1            # jump(s)
