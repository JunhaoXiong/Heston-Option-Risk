r"""Characteristic function of the Heston log-forward.

phi(u; T) = E[ exp(i u X_T) ],  X_T = ln(S_T / F_T),  F_T = S_0 e^{(r-q)T}.

Two algebraically identical forms are implemented:

* ``cf_little_trap`` (default) -- Albrecher, Mayer, Schoutens & Tistaert (2007),
  "The little Heston trap". Uses g2 = (b - d)/(b + d) and e^{-dT}.
* ``cf_heston1993`` -- the original form of Heston (1993) with
  g1 = (b + d)/(b - d) and e^{+dT}.

Both contain a complex logarithm ln((1 - g e^{...})/(1 - g)). Numerical
libraries return the *principal branch* of ln. Along the integration path
u in [0, inf) the argument of the original form can wind around the origin
(cross the negative real axis), so the principal-branch log jumps by 2*pi*i and
phi becomes discontinuous -> wrong prices, typically for long maturities and
large vol-of-vol. In the little-trap form the argument never crosses the
branch cut (proved by Albrecher et al. for Re d > 0), so the principal log is
the correct continuous branch without any rotation-counting. This is a
numerical-stability choice; it does not change the model or its PDE.
"""
from __future__ import annotations

import numpy as np

from heston_risk.heston.params import HestonParams


def _common(u, T, p: HestonParams):
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(b * b + p.sigma**2 * (iu + u * u))  # principal sqrt => Re d >= 0
    return u, iu, b, d


def cf_little_trap(u, T: float, p: HestonParams):
    """phi(u) for complex u (vectorised). Stable form (Albrecher et al. 2007)."""
    u, iu, b, d = _common(u, T, p)
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    s2 = p.sigma**2
    C = p.kappa * p.theta / s2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / s2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * p.v0)


def cf_heston1993(u, T: float, p: HestonParams):
    """phi(u) in the original Heston (1993) form. Kept only to demonstrate the
    branch-cut problem; do not use for pricing."""
    u, iu, b, d = _common(u, T, p)
    g = (b + d) / (b - d)
    e = np.exp(d * T)
    s2 = p.sigma**2
    C = p.kappa * p.theta / s2 * ((b + d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b + d) / s2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * p.v0)
