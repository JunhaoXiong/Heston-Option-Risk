r"""European option prices under Heston.

Primary method -- Lewis (2001) single-integral formula on the forward:

    C = e^{-rT} [ F - sqrt(F K)/pi * int_0^inf Re( e^{i u k} phi(u - i/2) ) / (u^2 + 1/4) du ],
    k = ln(F / K),   phi = CF of ln(S_T / F)  (little-trap form).

The integrand is smooth, bounded at u = 0 and decays like |phi(u - i/2)| / u^2,
so a composite Gauss-Legendre rule on a truncated range [0, U] is accurate and
fully vectorised across strikes (one CF evaluation per node per maturity).
U is chosen per maturity from the CF envelope (independent of the strike,
since |e^{iuk}| = 1).

Puts follow from put-call parity (exact for the same integral).

Two reference implementations for validation:
  * ``heston_price_quad``        -- same Lewis integral with adaptive QUADPACK on [0, inf).
  * ``heston_price_gil_pelaez``  -- Heston's original two-probability formula
                                    (independent formula, puts priced directly).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from heston_risk.bs import _is_call
from heston_risk.heston.charfunc import cf_little_trap
from heston_risk.heston.params import HestonParams

_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl(n: int):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def truncation_bound(T: float, p: HestonParams, tol: float = 1e-14, u_max: float = 1e5) -> float:
    """Smallest U on a log grid beyond which |phi(u - i/2)| / (u^2 + 1/4) < tol."""
    grid = np.geomspace(1e-2, u_max, 600)
    env = np.abs(cf_little_trap(grid - 0.5j, T, p)) / (grid**2 + 0.25)
    above = np.nonzero(env >= tol)[0]
    if above.size == 0:
        return float(grid[0])
    i = above[-1]
    return float(grid[min(i + 1, grid.size - 1)])


def _prep(S, K, T, r, q, kind):
    S, K, T, r, q = np.broadcast_arrays(*map(lambda a: np.asarray(a, float), (S, K, T, r, q)))
    call = np.broadcast_to(_is_call(kind), S.shape)
    if np.any(T <= 0):
        raise ValueError("T must be > 0")
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be > 0")
    return S, K, T, r, q, call


def _finish(call_px, S, K, T, r, q, call):
    df = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    put_px = call_px - df * (F - K)
    out = np.where(call, call_px, put_px)
    return out[()] if out.ndim == 0 else out


def _composite_gl(U: float, panel_width: float, nodes_per_panel: int):
    """Nodes/weights of composite Gauss-Legendre on [0, U].

    Panels are graded near the origin (edges 0, 1/4, 1/2, 1, 2, 4, ...) because
    the kernel 1/(u^2 + 1/4) and the CF vary on an O(1) scale there, then
    uniform of width ``panel_width`` out to U.
    """
    x, w = _gl(nodes_per_panel)
    geo = [0.0] + [0.25 * 2.0**j for j in range(int(np.log2(panel_width / 0.25)) + 1)]
    geo = [e for e in geo if e < U]
    start = geo[-1]
    m = max(1, int(np.ceil((U - start) / panel_width)))
    edges = np.concatenate([geo[:-1], np.linspace(start, U, m + 1)])
    h = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    u = (mid[:, None] + 0.5 * h[:, None] * x[None, :]).ravel()
    wu = (0.5 * h[:, None] * w[None, :]).ravel()
    return u, wu


def heston_price(S, K, T, r, q, params: HestonParams, kind="call",
                 panel_width: float = 20.0, nodes_per_panel: int = 24, tol: float = 1e-14):
    """Vectorised Heston European price (Lewis formula, composite Gauss-Legendre).

    All of S, K, T, r, q, kind broadcast together. The CF is evaluated once per
    distinct maturity on [0, U(T)] split into panels of width ``panel_width``
    with ``nodes_per_panel`` Gauss-Legendre nodes each. Short maturities have a
    slowly decaying CF (large U) and automatically get more panels.
    """
    S, K, T, r, q, call = _prep(S, K, T, r, q, kind)
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)
    integral = np.empty(S.shape)
    for t in np.unique(T):
        m = T == t
        U = truncation_bound(float(t), params, tol=tol)
        u, wu = _composite_gl(U, panel_width, nodes_per_panel)
        phi = cf_little_trap(u - 0.5j, float(t), params)            # (n,)
        kern = np.exp(1j * np.outer(k[m], u)) * phi / (u * u + 0.25)  # (m, n)
        integral[m] = kern.real @ wu
    call_px = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)
    return _finish(call_px, S, K, T, r, q, call)


def heston_price_quad(S, K, T, r, q, params: HestonParams, kind="call", epsabs=1e-13, epsrel=1e-12):
    """Reference: Lewis integral with adaptive quadrature on [0, inf). Slow."""
    S, K, T, r, q, call = _prep(S, K, T, r, q, kind)
    F = S * np.exp((r - q) * T)
    call_px = np.empty(S.shape)
    for idx in np.ndindex(S.shape):
        t, k = float(T[idx]), float(np.log(F[idx] / K[idx]))
        f = lambda u: (np.exp(1j * u * k) * cf_little_trap(u - 0.5j, t, params)).real / (u * u + 0.25)  # noqa: E731
        val, _ = quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=2000)
        call_px[idx] = np.exp(-r[idx] * t) * (F[idx] - np.sqrt(F[idx] * K[idx]) / np.pi * val)
    return _finish(call_px, S, K, T, r, q, call)


def heston_probabilities(F, K, T, params: HestonParams, epsabs=1e-13, epsrel=1e-12):
    """Gil-Pelaez probabilities (P1 under the stock measure, P2 = Q(S_T > K))."""
    k = float(np.log(F / K))

    def integrand(u, shift):
        return (np.exp(1j * u * k) * cf_little_trap(u - shift, T, params) / (1j * u)).real

    I1, _ = quad(integrand, 1e-12, np.inf, args=(1j,), epsabs=epsabs, epsrel=epsrel, limit=2000)
    I2, _ = quad(integrand, 1e-12, np.inf, args=(0.0,), epsabs=epsabs, epsrel=epsrel, limit=2000)
    return 0.5 + I1 / np.pi, 0.5 + I2 / np.pi


def heston_price_gil_pelaez(S, K, T, r, q, params: HestonParams, kind="call"):
    """Reference: Heston (1993) two-probability formula; puts priced directly
    (not via parity), so parity is a genuine cross-check."""
    S, K, T, r, q, call = _prep(S, K, T, r, q, kind)
    out = np.empty(S.shape)
    for idx in np.ndindex(S.shape):
        t = float(T[idx])
        df = np.exp(-r[idx] * t)
        F = S[idx] * np.exp((r[idx] - q[idx]) * t)
        P1, P2 = heston_probabilities(F, K[idx], t, params)
        if call[idx]:
            out[idx] = df * (F * P1 - K[idx] * P2)
        else:
            out[idx] = df * (K[idx] * (1.0 - P2) - F * (1.0 - P1))
    return out[()] if out.ndim == 0 else out
