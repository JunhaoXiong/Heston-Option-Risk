"""Black-Scholes(-Merton) prices and implied volatility.

Conventions used throughout the package:
  S  spot, K strike, T year fraction, r continuously-compounded risk-free rate,
  q continuous dividend yield, forward F = S exp((r - q) T).
  ``kind`` is "call" or "put" (array-broadcastable via ``is_call``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def _is_call(kind) -> np.ndarray:
    k = np.asarray(kind)
    if k.dtype.kind in "US":
        bad = ~np.isin(k, ["call", "put"])
        if np.any(bad):
            raise ValueError(f"kind must be 'call' or 'put', got {k[bad]}")
        return k == "call"
    return k.astype(bool)


def bs_price(S, K, T, r, q, sigma, kind="call"):
    """Vectorised Black-Scholes-Merton price.

    sigma = 0 or T = 0 returns the discounted intrinsic value on the forward.
    """
    S, K, T, r, q, sigma = np.broadcast_arrays(*map(lambda a: np.asarray(a, float), (S, K, T, r, q, sigma)))
    call = np.broadcast_to(_is_call(kind), S.shape)
    df = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    w = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * w**2) / w
        d2 = d1 - w
        c = df * (F * ndtr(d1) - K * ndtr(d2))
    intrinsic = df * np.maximum(F - K, 0.0)
    c = np.where(w > 0, c, intrinsic)
    p = c - df * (F - K)  # put-call parity
    out = np.where(call, c, p)
    return out[()] if out.ndim == 0 else out


def bs_vega(S, K, T, r, q, sigma):
    """dPrice/dsigma (same for calls and puts)."""
    S, K, T, r, q, sigma = map(lambda a: np.asarray(a, float), (S, K, T, r, q, sigma))
    F = S * np.exp((r - q) * T)
    w = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * w**2) / w
    return np.exp(-r * T) * F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(T)


def implied_vol(price, S, K, T, r, q, kind="call", lo=1e-6, hi=5.0, xtol=1e-12):
    """Implied volatility by Brent root-finding, element-wise.

    Returns NaN where the price violates the no-arbitrage bounds
    (below discounted intrinsic or above the upper bound) or where the
    root is not bracketed by [lo, hi]. Never silently clips.
    """
    price, S, K, T, r, q = np.broadcast_arrays(*map(lambda a: np.asarray(a, float), (price, S, K, T, r, q)))
    call = np.broadcast_to(_is_call(kind), price.shape)
    out = np.full(price.shape, np.nan)
    for idx in np.ndindex(price.shape):
        p, s, k, t, rr, qq, c = price[idx], S[idx], K[idx], T[idx], r[idx], q[idx], bool(call[idx])
        kind_i = "call" if c else "put"
        df, F = np.exp(-rr * t), s * np.exp((rr - qq) * t)
        lower = df * max(F - k, 0.0) if c else df * max(k - F, 0.0)
        upper = df * F if c else df * k
        if not (lower < p < upper) or t <= 0:
            continue
        f = lambda v: bs_price(s, k, t, rr, qq, v, kind_i) - p  # noqa: E731
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            continue
        out[idx] = brentq(f, lo, hi, xtol=xtol, maxiter=200)
    return out[()] if out.ndim == 0 else out
