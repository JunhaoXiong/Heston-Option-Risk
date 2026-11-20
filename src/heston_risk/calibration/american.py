"""Early-exercise premium check for American SPY options (CRR binomial tree).

SPY options are American. Calibration uses only out-of-the-money quotes treated as
European; this module measures how large the early-exercise premium is for those
quotes at their own implied volatility, with a continuous dividend yield q implied
from the forward (F = S e^{(r-q)T}).
"""
from __future__ import annotations

import numpy as np

from heston_risk.bs import bs_price


def crr_price(S, K, T, r, q, sigma, kind="put", american=True, n=500):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    pu = (np.exp((r - q) * dt) - d) / (u - d)
    j = np.arange(n + 1)
    ST = S * u ** (n - j) * d ** j
    call = kind == "call"
    V = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    for i in range(n - 1, -1, -1):
        V = disc * (pu * V[:-1] + (1 - pu) * V[1:])
        if american:
            Si = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            V = np.maximum(V, (Si - K) if call else (K - Si))
    return float(V[0])


def early_exercise_premium(S, K, T, r, q, sigma, kind, n=500):
    """American minus European CRR price (same tree, so discretisation error cancels),
    together with the European BS price for scale."""
    am = crr_price(S, K, T, r, q, sigma, kind, True, n)
    eu = crr_price(S, K, T, r, q, sigma, kind, False, n)
    return am - eu, float(bs_price(S, K, T, r, q, sigma, kind))


def de_americanize(mid, S, K, T, r, q, kind, iv_guess, n=300):
    """European-equivalent price of an American quote.

    Solve sigma_A with CRR_American(sigma_A) = mid, then return
    mid - (CRR_American(sigma_A) - CRR_European(sigma_A)), i.e. the quote minus the
    early-exercise premium at the volatility that reproduces it. Using the same tree
    for both terms cancels most of the tree discretisation error.
    Returns (european_equivalent_price, sigma_A); NaN if no root is bracketed.
    """
    from scipy.optimize import brentq

    f = lambda s: crr_price(S, K, T, r, q, s, kind, True, n) - mid  # noqa: E731
    lo, hi = max(1e-3, 0.5 * iv_guess), 2.0 * iv_guess
    if f(lo) * f(hi) > 0:
        lo, hi = 1e-3, 3.0
        if f(lo) * f(hi) > 0:
            return float("nan"), float("nan")
    sA = brentq(f, lo, hi, xtol=1e-7)
    prem = crr_price(S, K, T, r, q, sA, kind, True, n) - crr_price(S, K, T, r, q, sA, kind, False, n)
    return float(mid - prem), float(sA)
