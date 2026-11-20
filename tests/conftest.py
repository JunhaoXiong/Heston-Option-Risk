import numpy as np
import pytest

from heston_risk import HestonParams

# Fang & Oosterlee (2008), "A novel pricing method for European options based on
# Fourier-cosine series expansions", SIAM J. Sci. Comput. 31(2), Heston test case.
FO_PARAMS = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
FO_REFERENCE_CALL = 5.785155450  # S0 = K = 100, T = 1, r = q = 0 (as published)
# Same option re-computed with mpmath at 30 significant digits (scripts/run_phase1.py, section 1).
# It differs from the published value by 1.6e-8, so tests use this as the tight reference.
FO_HIGH_PRECISION_CALL = 5.785155434376189


@pytest.fixture
def base_params():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)


def random_cases(n, seed=0):
    """Random (params, T, K) covering short/long maturities and wide strikes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = HestonParams(v0=rng.uniform(0.005, 0.2), kappa=rng.uniform(0.2, 6), theta=rng.uniform(0.005, 0.2),
                         sigma=rng.uniform(0.05, 1.5), rho=rng.uniform(-0.95, 0.5))
        T = float(np.exp(rng.uniform(np.log(1 / 52), np.log(5))))
        K = float(100 * np.exp(rng.uniform(-0.6, 0.6) * np.sqrt(T)))
        out.append((p, T, K))
    return out
