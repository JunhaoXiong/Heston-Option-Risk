"""Full-truncation Euler scheme (Lord, Koekkoek & van Dijk 2010).

Phase 1 uses it only as an independent sanity check of the analytic pricer;
Phase 2 adds the Andersen (2008) QE scheme and a proper bias/variance study.

    v+    = max(v, 0)
    ln S += (r - q - v+/2) dt + sqrt(v+ dt) Z_S
    v    += kappa (theta - v+) dt + sigma sqrt(v+ dt) Z_v,   corr(Z_S, Z_v) = rho
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heston_risk.heston.params import HestonParams


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int
    n_steps: int


def simulate_terminal_euler(S0, T, r, q, p: HestonParams, n_paths: int, n_steps: int,
                            rng: np.random.Generator, antithetic: bool = True):
    """Terminal S_T for ``n_paths`` paths (antithetic pairs if requested)."""
    half = n_paths // 2 if antithetic else n_paths
    dt = T / n_steps
    sq = np.sqrt(1.0 - p.rho**2)
    lnS = np.full(half * (2 if antithetic else 1), np.log(S0))
    v = np.full_like(lnS, p.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        if antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
        zv = z1
        zs = p.rho * z1 + sq * z2
        vp = np.maximum(v, 0.0)
        sd = np.sqrt(vp * dt)
        lnS += (r - q - 0.5 * vp) * dt + sd * zs
        v += p.kappa * (p.theta - vp) * dt + p.sigma * sd * zv
    return np.exp(lnS)


def price_euler_mc(S0, K, T, r, q, p: HestonParams, kind="call", n_paths=200_000, n_steps=200,
                   seed=12345) -> MCResult:
    """Discounted MC price with standard error (antithetic pairs averaged first)."""
    rng = np.random.default_rng(seed)
    ST = simulate_terminal_euler(S0, T, r, q, p, n_paths, n_steps, rng, antithetic=True)
    pay = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    half = ST.size // 2
    pair = 0.5 * (pay[:half] + pay[half:])  # independent samples
    df = np.exp(-r * T)
    return MCResult(price=float(df * pair.mean()), std_error=float(df * pair.std(ddof=1) / np.sqrt(half)),
                    n_paths=ST.size, n_steps=n_steps)
