r"""Training data for the surrogate: (Heston parameters, log-moneyness, maturity) -> implied vol.

Why the problem has only 7 inputs: pricing on the forward with F = 1 and r = 0 loses nothing,
because the Black-Scholes implied vol of a Heston price depends only on
(v0, kappa, theta, sigma, rho, k = ln(K/F), T) -- spot level, rates and dividends enter only
through F and the discount factor, which cancel when converting price -> IV.

Sampling design
* A "slice" = one parameter vector and one maturity, drawn from a scrambled Sobol sequence
  over the agreed box (T log-uniform, so short maturities get more samples).
* Each slice gets ``strikes_per_slice`` log-moneyness values uniform in +-0.6 sqrt(T).
  One CF evaluation then prices the whole slice (fast), and train/validation/test are
  split by slice, never by row, so near-duplicate points of one slice cannot leak
  between training and test.
* Labels: out-of-the-money option (put for k < 0, call for k >= 0) priced by the Phase 1
  analytic pricer, then inverted to an IV with Brent. Points whose price is too small to
  invert reliably (time value < 1e-9 of the forward) are dropped and counted.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from heston_risk.bs import implied_vol
from heston_risk.heston.params import HestonParams
from heston_risk.heston.pricer import heston_price

PARAM_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
FEATURES = PARAM_NAMES + ("k", "T")
BOX_LO = np.array([0.005, 0.1, 0.005, 0.1, -0.95])
BOX_HI = np.array([0.25, 10.0, 0.15, 3.0, 0.0])
T_MIN, T_MAX = 7 / 365, 2.5
K_WIDTH = 0.6
MIN_PRICE = 1e-9


def sample_slices(n_slices: int, seed: int) -> np.ndarray:
    """(n, 6) array of [v0, kappa, theta, sigma, rho, T] from a scrambled Sobol sequence."""
    u = qmc.Sobol(d=6, scramble=True, seed=seed).random(n_slices)
    p = BOX_LO + u[:, :5] * (BOX_HI - BOX_LO)
    T = np.exp(np.log(T_MIN) + u[:, 5] * (np.log(T_MAX) - np.log(T_MIN)))
    return np.column_stack([p, T])


def label_slice(row: np.ndarray, ks: np.ndarray):
    """Return (iv, price) for one slice; NaN iv where not invertible."""
    v0, kappa, theta, sigma, rho, T = row
    p = HestonParams(v0, kappa, theta, sigma, rho)
    K = np.exp(ks)
    kind = np.where(ks < 0, "put", "call")
    px = heston_price(1.0, K, T, 0.0, 0.0, p, kind)
    iv = np.full(ks.shape, np.nan)
    ok = px > MIN_PRICE
    if ok.any():
        iv[ok] = implied_vol(px[ok], 1.0, K[ok], T, 0.0, 0.0, kind[ok])
    return iv, px


def _work(args):
    rows, ks = args
    out = np.empty(ks.shape)
    for i in range(rows.shape[0]):
        out[i], _ = label_slice(rows[i], ks[i])
    return out


def make_dataset(n_slices: int, strikes_per_slice: int, seed: int, workers: int | None = None):
    """Return dict with X (n_rows, 7), y (n_rows,), slice_id (n_rows,), n_dropped."""
    rows = sample_slices(n_slices, seed)
    rng = np.random.default_rng(seed + 1)
    width = K_WIDTH * np.sqrt(rows[:, 5])
    ks = (rng.uniform(-1, 1, (n_slices, strikes_per_slice))) * width[:, None]
    workers = workers or os.cpu_count() or 1
    chunks = np.array_split(np.arange(n_slices), workers * 8)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_work, [(rows[c], ks[c]) for c in chunks]))
    iv = np.concatenate(parts, axis=0)
    X = np.column_stack([np.repeat(rows[:, :5], strikes_per_slice, axis=0), ks.ravel(),
                         np.repeat(rows[:, 5], strikes_per_slice)])
    y = iv.ravel()
    sid = np.repeat(np.arange(n_slices), strikes_per_slice)
    keep = np.isfinite(y)
    return {"X": X[keep].astype(np.float64), "y": y[keep], "slice_id": sid[keep],
            "n_dropped": int((~keep).sum()), "n_total": int(y.size)}
