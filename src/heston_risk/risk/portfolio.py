r"""Option positions, Heston valuation of a book, and finite-difference Greeks.

State variables: spot S and instantaneous variance v (the Heston v0). kappa, theta, sigma, rho
are held fixed ("sticky parameters"). Each option carries its own dividend yield q, implied from
its expiry's forward (F = S e^{(r-q)T}), so the book reprices consistently with the market forwards.

Greeks (central finite differences on the analytic pricer, per unit of the book):
  delta = dV/dS,  gamma = d2V/dS2,  vega_v = dV/dv,  vanna = d2V/dS dv,  volga = d2V/dv2,
  theta = dV/dt (value change per year of calendar time passing, i.e. -dV/dT).
For reading convenience vega is also reported per volatility point: dV/d(sqrt v) * 0.01.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heston_risk.heston.params import HestonParams
from heston_risk.heston.pricer import heston_price

MULTIPLIER = 100.0


@dataclass(frozen=True)
class Option:
    kind: str        # "call" or "put"
    K: float
    T: float         # years to expiry at the valuation date
    q: float         # dividend yield implied from this expiry's forward
    qty: float       # contracts, + long / - short
    label: str = ""


@dataclass(frozen=True)
class Book:
    options: tuple
    shares: float = 0.0      # SPY shares (hedge)
    name: str = ""


@dataclass(frozen=True)
class Market:
    S: float
    v: float
    r: float
    params: HestonParams     # kappa, theta, sigma, rho used; params.v0 is ignored in favour of v


def _p(m: Market, v: float) -> HestonParams:
    p = m.params
    return HestonParams(v0=max(v, 0.0), kappa=p.kappa, theta=p.theta, sigma=p.sigma, rho=p.rho)


def option_values(book: Book, m: Market, S=None, v=None, dt: float = 0.0) -> np.ndarray:
    """Value of each option position (with multiplier) at state (S, v) after time dt has passed."""
    S = m.S if S is None else S
    v = m.v if v is None else v
    p = _p(m, v)
    out = np.empty(len(book.options))
    for i, o in enumerate(book.options):
        T = o.T - dt
        if T <= 1e-10:
            intrinsic = max(S - o.K, 0.0) if o.kind == "call" else max(o.K - S, 0.0)
            out[i] = o.qty * MULTIPLIER * intrinsic
        else:
            out[i] = o.qty * MULTIPLIER * float(heston_price(S, o.K, T, m.r, o.q, p, o.kind))
    return out


def book_value(book: Book, m: Market, S=None, v=None, dt: float = 0.0) -> float:
    S_ = m.S if S is None else S
    return float(option_values(book, m, S, v, dt).sum() + book.shares * S_)


def greeks(book: Book, m: Market, hS_rel: float = 1e-3, hv_rel: float = 1e-2, ht: float = 1 / 365) -> dict:
    S, v = m.S, m.v
    hS, hv = hS_rel * S, hv_rel * v
    def V(s=S, v=v, dt=0.0):
        return book_value(book, m, s, v, dt)

    V0 = V()
    Vu, Vd = V(S + hS), V(S - hS)
    Vvu, Vvd = V(v=v + hv), V(v=v - hv)
    Vuu, Vud, Vdu, Vdd = V(S + hS, v + hv), V(S + hS, v - hv), V(S - hS, v + hv), V(S - hS, v - hv)
    g = {
        "value": V0,
        "delta": (Vu - Vd) / (2 * hS),
        "gamma": (Vu - 2 * V0 + Vd) / hS**2,
        "vega_v": (Vvu - Vvd) / (2 * hv),
        "vanna": (Vuu - Vud - Vdu + Vdd) / (4 * hS * hv),
        "volga": (Vvu - 2 * V0 + Vvd) / hv**2,
        "theta": (V(dt=ht) - V0) / ht,
    }
    g["vega_per_volpt"] = g["vega_v"] * 2.0 * np.sqrt(v) * 0.01   # dV/dsqrt(v) * 1 vol point
    return g


def delta_hedge_shares(book: Book, m: Market) -> float:
    """Shares that make the option book delta-neutral."""
    opts_only = Book(book.options, 0.0, book.name)
    return -greeks(opts_only, m)["delta"]
