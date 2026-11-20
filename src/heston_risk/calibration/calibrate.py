"""Heston calibration to an implied-volatility surface.

Objective: vega-weighted price residuals  e_i = (C_model_i - C_mid_i) / vega_i,
which to first order equal the implied-vol errors (sigma_model_i - sigma_mkt_i), so the
least-squares fit is approximately an IV-RMSE fit at the cost of price evaluations only.

Optimiser: scipy.optimize.least_squares (trust-region reflective) inside box bounds,
from several Latin-hypercube starting points, because the Heston objective is
non-convex and has flat valleys (e.g. kappa and sigma trade off against each other).

Feller-constrained variant: optimise f in (0, 1] instead of sigma, with
sigma = sqrt(2 kappa theta f), so 2 kappa theta >= sigma^2 holds by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from heston_risk.bs import implied_vol
from heston_risk.heston.params import HestonParams
from heston_risk.heston.pricer import heston_price

NAMES = ("v0", "kappa", "theta", "sigma", "rho")
LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
UPPER = np.array([1.0, 20.0, 1.0, 5.0, 0.999])
FELLER_LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-4, -0.999])   # 4th coordinate is f in (0, 1]
FELLER_UPPER = np.array([1.0, 20.0, 1.0, 1.0, 0.999])


def _to_params(x, feller: bool) -> HestonParams:
    v0, kappa, theta, s, rho = x
    sigma = np.sqrt(2.0 * kappa * theta * s) if feller else s
    return HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=max(sigma, 1e-8), rho=rho)


def model_prices(quotes: pd.DataFrame, p: HestonParams, r: float) -> np.ndarray:
    return heston_price(quotes.S_eff.values, quotes.strike.values, quotes["T"].values, r, 0.0, p,
                        quotes.kind.values.astype(str))


def residuals(x, quotes: pd.DataFrame, r: float, feller: bool) -> np.ndarray:
    p = _to_params(x, feller)
    return (model_prices(quotes, p, r) - quotes.mid.values) / quotes.vega.values


@dataclass
class CalibrationResult:
    params: HestonParams
    cost: float                  # 0.5 * sum of squared vega-weighted residuals
    rmse_vega_weighted: float    # approx IV RMSE (vol units)
    n_quotes: int
    feller_constrained: bool
    starts: list = field(default_factory=list)   # (x0, x_hat, cost, nfev, success) for every start
    seconds: float = 0.0


def calibrate(quotes: pd.DataFrame, r: float, n_starts: int = 10, seed: int = 0, feller: bool = False,
              max_nfev: int = 400) -> CalibrationResult:
    lo, hi = (FELLER_LOWER, FELLER_UPPER) if feller else (LOWER, UPPER)
    # starting points: Latin hypercube over a "plausible" sub-box (not the whole bound box)
    s_lo = np.array([0.005, 0.2, 0.005, 0.1 if not feller else 0.05, -0.95])
    s_hi = np.array([0.15, 8.0, 0.15, 1.5 if not feller else 1.0, -0.1])
    sob = qmc.LatinHypercube(d=5, seed=seed).random(n_starts)
    x0s = qmc.scale(sob, s_lo, s_hi)
    t0 = time.perf_counter()
    starts = []
    best = None
    for x0 in x0s:
        res = least_squares(residuals, x0, bounds=(lo, hi), args=(quotes, r, feller), method="trf",
                            x_scale="jac", max_nfev=max_nfev)
        starts.append((x0, res.x, float(res.cost), int(res.nfev), bool(res.success)))
        if best is None or res.cost < best.cost:
            best = res
    secs = time.perf_counter() - t0
    n = len(quotes)
    return CalibrationResult(_to_params(best.x, feller), float(best.cost), float(np.sqrt(2 * best.cost / n)), n,
                             feller, starts, secs)


def fit_report(quotes: pd.DataFrame, p: HestonParams, r: float) -> pd.DataFrame:
    """Per-quote model price, model IV and errors."""
    q = quotes.copy()
    q["model"] = model_prices(q, p, r)
    q["model_iv"] = implied_vol(q.model.values, q.S_eff.values, q.strike.values, q["T"].values, r, 0.0,
                                q.kind.values.astype(str))
    q["iv_err"] = q.model_iv - q.iv
    q["in_spread"] = (q.model >= q.bid) & (q.model <= q.ask)
    return q


def fit_black_scholes(quotes: pd.DataFrame) -> float:
    """Benchmark: the single constant volatility minimising the IV RMSE (= mean IV)."""
    return float(quotes.iv.mean())
