r"""Scenario P&L by five revaluation methods, and VaR / ES.

Scenario = (dS, dv) applied to today's state over horizon h (years), from historical
(SPY log return, VIX log change) pairs:  S' = S e^{x},  v' = v (VIX_{t+h} / VIX_t)^2.

Methods
  delta        : Delta dS + Theta h
  delta_gamma  : + 1/2 Gamma dS^2
  dgv          : + Vega_v dv + Vanna dS dv + 1/2 Volga dv^2        (second-order Taylor in (S, v))
  full         : exact Heston revaluation of every option at (S', v', T - h)      (benchmark)
  full_nn      : full revaluation with the Phase 4 MLP surrogate (IV -> Black-Scholes price);
                 scenarios outside the surrogate's training box fall back to the analytic pricer
                 and are counted.
VaR_a = a-quantile of the loss L = -P&L; ES_a = mean of losses at or beyond VaR_a.
"""
from __future__ import annotations

import time

import numpy as np

from heston_risk.bs import bs_price
from heston_risk.risk.portfolio import MULTIPLIER, Book, Market, book_value, greeks


def historical_scenarios(spy: np.ndarray, vix: np.ndarray, horizon_days: int):
    """Overlapping h-day (log SPY return, VIX ratio) pairs."""
    x = np.log(spy[horizon_days:] / spy[:-horizon_days])
    vr = vix[horizon_days:] / vix[:-horizon_days]
    return x, vr


def var_es(pnl: np.ndarray, var_level=0.99, es_level=0.975):
    loss = -np.asarray(pnl)
    var = float(np.quantile(loss, var_level))
    q_es = np.quantile(loss, es_level)
    es = float(loss[loss >= q_es].mean())
    return var, es


def taylor_pnl(g: dict, dS, dv, h, order: str):
    pnl = g["delta"] * dS + g["theta"] * h
    if order in ("delta_gamma", "dgv"):
        pnl = pnl + 0.5 * g["gamma"] * dS**2
    if order == "dgv":
        pnl = pnl + g["vega_v"] * dv + g["vanna"] * dS * dv + 0.5 * g["volga"] * dv**2
    return pnl


def full_pnl(book: Book, m: Market, S_new, v_new, h):
    V0 = book_value(book, m)
    return np.array([book_value(book, m, s, v, h) for s, v in zip(S_new, v_new)]) - V0


def nn_book_values(book: Book, m: Market, S_new, v_new, h, surrogate):
    """Book values with the surrogate (IV -> Black-Scholes); returns (values, n_fallback).

    Option evaluations outside the surrogate's training box use the analytic pricer instead.
    """
    from heston_risk.heston.params import HestonParams
    from heston_risk.heston.pricer import heston_price
    from heston_risk.surrogate.model import in_domain

    S_new, v_new = np.atleast_1d(np.asarray(S_new, float)), np.atleast_1d(np.asarray(v_new, float))
    p = m.params
    n = len(S_new)
    total = book.shares * S_new
    fallback = 0
    for o in book.options:
        T = o.T - h
        F = S_new * np.exp((m.r - o.q) * T)
        X = np.column_stack([v_new, np.full(n, p.kappa), np.full(n, p.theta), np.full(n, p.sigma),
                             np.full(n, p.rho), np.log(o.K / F), np.full(n, T)])
        ok = in_domain(X)
        px = np.empty(n)
        if ok.any():
            px[ok] = bs_price(S_new[ok], o.K, T, m.r, o.q, surrogate.iv(X[ok]), o.kind)
        for i in np.nonzero(~ok)[0]:
            pp = HestonParams(max(v_new[i], 0.0), p.kappa, p.theta, p.sigma, p.rho)
            px[i] = float(heston_price(S_new[i], o.K, T, m.r, o.q, pp, o.kind))
        fallback += int((~ok).sum())
        total = total + o.qty * MULTIPLIER * px
    return total, fallback


def full_pnl_nn(book: Book, m: Market, S_new, v_new, h, surrogate):
    """Full revaluation P&L with the surrogate. Today's value is also taken from the surrogate, so a
    constant pricing bias of the network cancels in the P&L instead of showing up as a fake loss."""
    base, _ = nn_book_values(book, m, [m.S], [m.v], 0.0, surrogate)
    vals, fb = nn_book_values(book, m, S_new, v_new, h, surrogate)
    return vals - base[0], fb


def run_methods(book: Book, m: Market, x, vix_ratio, h, surrogate=None):
    S_new = m.S * np.exp(x)
    v_new = m.v * vix_ratio**2
    dS, dv = S_new - m.S, v_new - m.v
    g = greeks(book, m)
    out, secs = {}, {}
    for name in ("delta", "delta_gamma", "dgv"):
        t0 = time.perf_counter()
        out[name] = taylor_pnl(g, dS, dv, h, name)
        secs[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["full"] = full_pnl(book, m, S_new, v_new, h)
    secs["full"] = time.perf_counter() - t0
    fb = None
    if surrogate is not None:
        t0 = time.perf_counter()
        out["full_nn"], fb = full_pnl_nn(book, m, S_new, v_new, h, surrogate)
        secs["full_nn"] = time.perf_counter() - t0
    return out, secs, g, fb
