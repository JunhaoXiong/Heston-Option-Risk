r"""Monte Carlo simulation of the Heston model: full-truncation Euler and Andersen QE.

Both schemes advance (ln S, v) over a step Delta with the same random inputs:
    z_v  standard normal driving the variance (QE: also mapped to a uniform U = Phi(z_v)),
    z_x  standard normal, independent of z_v, driving the part of the price shock
         orthogonal to the variance shock.
Antithetic pairs flip the sign of both (U -> 1 - U).

Full-truncation Euler (Lord, Koekkoek & van Dijk 2010)
    v+     = max(v, 0)
    ln S' = ln S + (r - q - v+/2) D + sqrt(v+ D) (rho z_v + sqrt(1 - rho^2) z_x)
    v'     = v + kappa (theta - v+) D + sigma sqrt(v+ D) z_v

Andersen (2008) quadratic-exponential (QE) scheme
  Variance: match the first two conditional moments of v' | v (exact CIR moments)
      m   = theta + (v - theta) e^{-kappa D}
      s^2 = v sigma^2 e^{-kappa D} (1 - e^{-kappa D}) / kappa + theta sigma^2 (1 - e^{-kappa D})^2 / (2 kappa)
      psi = s^2 / m^2
    psi <= psi_c (quadratic):  v' = a (b + z_v)^2,
        b^2 = 2/psi - 1 + sqrt(2/psi) sqrt(2/psi - 1),  a = m / (1 + b^2)
    psi >  psi_c (exponential, with an atom at 0):
        p = (psi - 1)/(psi + 1),  beta = (1 - p)/m,
        v' = 0 if U <= p else ln((1 - p)/(1 - U)) / beta
  Log-price: exact representation ln S' = ln S + (r-q)D + rho/sigma (v' - v - kappa theta D)
      + (kappa rho / sigma - 1/2) int v + sqrt(1 - rho^2) int sqrt(v) dW,
    with int_t^{t+D} v du ~ D (gamma1 v + gamma2 v') (central rule gamma1 = gamma2 = 1/2):
      ln S' = ln S + (r - q) D + K0 + K1 v + K2 v' + sqrt(K3 v + K4 v') z_x
      K0 = -rho kappa theta D / sigma
      K1 = gamma1 D (kappa rho / sigma - 1/2) - rho / sigma
      K2 = gamma2 D (kappa rho / sigma - 1/2) + rho / sigma
      K3 = gamma1 D (1 - rho^2),  K4 = gamma2 D (1 - rho^2)
  Martingale correction (Andersen sec. 4.3): replace K0 path-wise by K0* so that
      E[S' | S, v] = S e^{(r-q)D} holds exactly for the discrete scheme:
      K0* = -ln E[exp(A v') | v] - (K1 + K3/2) v,   A = K2 + K4/2,
      quadratic:   E[e^{A v'}] = exp(A b^2 a / (1 - 2 A a)) / sqrt(1 - 2 A a)   (needs A < 1/(2a))
      exponential: E[e^{A v'}] = p + beta (1 - p) / (beta - A)                (needs A < beta)
    Where the condition fails the plain K0 is used and the number of such cases is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from heston_risk.heston.params import HestonParams

SCHEMES = ("euler", "qe")


@dataclass
class StepStats:
    """Diagnostics accumulated over a simulation."""
    n_variance_draws: int = 0
    n_exponential_branch: int = 0       # QE: psi > psi_c
    n_martingale_fallback: int = 0      # QE: correction condition violated
    n_negative_variance: int = 0        # Euler: v' < 0 before truncation

    def merge(self, other: "StepStats") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass(frozen=True)
class QEConfig:
    psi_c: float = 1.5
    gamma1: float = 0.5
    gamma2: float = 0.5
    martingale_correction: bool = True


def _euler_step(x, v, dt, r, q, p: HestonParams, zv, zx, stats: StepStats):
    vp = np.maximum(v, 0.0)
    sd = np.sqrt(vp * dt)
    x_new = x + (r - q - 0.5 * vp) * dt + sd * (p.rho * zv + np.sqrt(1.0 - p.rho**2) * zx)
    v_new = v + p.kappa * (p.theta - vp) * dt + p.sigma * sd * zv
    stats.n_variance_draws += v.size
    stats.n_negative_variance += int(np.count_nonzero(v_new < 0))
    return x_new, v_new


def _qe_step(x, v, dt, r, q, p: HestonParams, zv, zx, stats: StepStats, cfg: QEConfig):
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho
    e = np.exp(-k * dt)
    m = th + (v - th) * e
    s2 = v * s * s * e * (1.0 - e) / k + th * s * s * (1.0 - e) ** 2 / (2.0 * k)
    psi = s2 / (m * m)

    quad = psi <= cfg.psi_c
    v_new = np.empty_like(v)
    # quadratic branch
    inv = 2.0 / psi[quad]
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
    a = m[quad] / (1.0 + b2)
    b = np.sqrt(b2)
    v_new[quad] = a * (b + zv[quad]) ** 2
    # exponential branch
    ex = ~quad
    pe = (psi[ex] - 1.0) / (psi[ex] + 1.0)
    beta = (1.0 - pe) / m[ex]
    U = ndtr(zv[ex])
    with np.errstate(divide="ignore"):
        v_new[ex] = np.where(U <= pe, 0.0, np.log((1.0 - pe) / (1.0 - U)) / beta)

    g1, g2 = cfg.gamma1, cfg.gamma2
    K0 = -rho * k * th * dt / s
    K1 = g1 * dt * (k * rho / s - 0.5) - rho / s
    K2 = g2 * dt * (k * rho / s - 0.5) + rho / s
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    if cfg.martingale_correction:
        A = K2 + 0.5 * K4
        logM = np.empty_like(v)
        ok = np.empty(v.shape, dtype=bool)
        # quadratic
        aq = a
        okq = A < 1.0 / (2.0 * aq)
        den = np.where(okq, 1.0 - 2.0 * A * aq, 1.0)
        logM[quad] = A * b2 * aq / den - 0.5 * np.log(den)
        ok[quad] = okq
        # exponential
        oke = A < beta
        logM[ex] = np.log(pe + beta * (1.0 - pe) / np.where(oke, beta - A, 1.0))
        ok[ex] = oke
        K0v = np.where(ok, -logM - (K1 + 0.5 * K3) * v, K0)
        stats.n_martingale_fallback += int(np.count_nonzero(~ok))
    else:
        K0v = K0

    x_new = x + (r - q) * dt + K0v + K1 * v + K2 * v_new + np.sqrt(K3 * v + K4 * v_new) * zx
    stats.n_variance_draws += v.size
    stats.n_exponential_branch += int(np.count_nonzero(ex))
    return x_new, v_new


def _normals(rng: np.random.Generator, half: int, antithetic: bool):
    zv = rng.standard_normal(half)
    zx = rng.standard_normal(half)
    if antithetic:
        zv = np.concatenate([zv, -zv])
        zx = np.concatenate([zx, -zx])
    return zv, zx


def simulate(S0, T, r, q, p: HestonParams, n_paths: int, n_steps: int, rng: np.random.Generator,
             scheme: str = "qe", antithetic: bool = True, qe: QEConfig = QEConfig(),
             save_every: int | None = None):
    """Simulate Heston paths.

    Returns (times, S, v, stats). With ``save_every=None`` only the terminal
    values are kept (S, v have shape (n_paths,), times = [T]); otherwise every
    ``save_every``-th step is stored, giving arrays of shape (n_saved + 1, n_paths)
    including t = 0. With antithetic=True, path i and path i + n_paths/2 are a pair.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic=True")
    half = n_paths // 2 if antithetic else n_paths
    dt = T / n_steps
    x = np.full(n_paths, np.log(S0), dtype=float)
    v = np.full(n_paths, p.v0, dtype=float)
    stats = StepStats()
    keep_t, keep_S, keep_v = [0.0], [np.full(n_paths, float(S0))], [v.copy()]
    for i in range(1, n_steps + 1):
        zv, zx = _normals(rng, half, antithetic)
        if scheme == "qe":
            x, v = _qe_step(x, v, dt, r, q, p, zv, zx, stats, qe)
        else:
            x, v = _euler_step(x, v, dt, r, q, p, zv, zx, stats)
        if save_every is not None and (i % save_every == 0 or i == n_steps):
            keep_t.append(i * dt)
            keep_S.append(np.exp(x))
            keep_v.append(v.copy())
    if save_every is None:
        return np.array([T]), np.exp(x), v, stats
    return np.array(keep_t), np.vstack(keep_S), np.vstack(keep_v), stats


@dataclass
class MCPrice:
    price: np.ndarray            # per strike
    std_error: np.ndarray
    n_paths: int
    n_steps: int
    scheme: str
    control_variate: bool
    stats: StepStats = field(default_factory=StepStats)
    seconds: float = 0.0


def price_mc(S0, K, T, r, q, p: HestonParams, kind="call", n_paths: int = 200_000, n_steps: int = 100,
             scheme: str = "qe", seed: int = 12345, antithetic: bool = True, control_variate: bool = False,
             batch_paths: int = 200_000, qe: QEConfig = QEConfig()) -> MCPrice:
    """European prices for several strikes from one set of paths.

    Independent samples are antithetic-pair averages (pairs are not independent
    of each other within a pair). With control_variate=True the discounted
    terminal stock e^{-rT} S_T, whose mean S0 e^{-qT} is known exactly, is used
    as a control with the optimal coefficient estimated from the same sample.
    """
    import time

    K = np.atleast_1d(np.asarray(K, float))
    is_call = np.broadcast_to(np.asarray(kind) == "call", K.shape)
    rng = np.random.default_rng(seed)
    df = np.exp(-r * T)
    ys, xs = [], []
    stats = StepStats()
    t0 = time.perf_counter()
    remaining = n_paths
    while remaining > 0:
        nb = min(batch_paths, remaining)
        nb -= nb % 2 if antithetic else 0
        _, ST, _, st = simulate(S0, T, r, q, p, nb, n_steps, rng, scheme, antithetic, qe)
        stats.merge(st)
        pay = np.where(is_call[:, None], np.maximum(ST[None, :] - K[:, None], 0.0),
                       np.maximum(K[:, None] - ST[None, :], 0.0)) * df
        ctrl = df * ST
        if antithetic:
            h = nb // 2
            pay = 0.5 * (pay[:, :h] + pay[:, h:])
            ctrl = 0.5 * (ctrl[:h] + ctrl[h:])
        ys.append(pay)
        xs.append(ctrl)
        remaining -= nb
    secs = time.perf_counter() - t0
    Y = np.concatenate(ys, axis=1)
    X = np.concatenate(xs)
    n = X.size
    if control_variate:
        mu_x = S0 * np.exp(-q * T)
        xc = X - X.mean()
        bstar = (Y - Y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
        Y = Y - bstar[:, None] * (X[None, :] - mu_x)
    price = Y.mean(axis=1)
    se = Y.std(axis=1, ddof=1) / np.sqrt(n)
    return MCPrice(price, se, n_paths, n_steps, scheme, control_variate, stats, secs)
