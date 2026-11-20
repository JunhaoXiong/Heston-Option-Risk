"""Heston parameter container.

Risk-neutral dynamics (r, q constant):
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t,   d<W^S, W^v>_t = rho dt
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    sigma: float   # vol of variance ("vol-of-vol")
    rho: float     # spot/variance correlation

    def __post_init__(self):
        if self.v0 < 0:
            raise ValueError("v0 must be >= 0")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if self.theta < 0:
            raise ValueError("theta must be >= 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0 (use the Black-Scholes limit for sigma = 0)")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2; >= 1 means v_t stays strictly positive."""
        return 2.0 * self.kappa * self.theta / self.sigma**2

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio >= 1.0

    def expected_avg_variance(self, T: float) -> float:
        """E[(1/T) int_0^T v_t dt] = theta + (v0 - theta)(1 - e^{-kappa T})/(kappa T)."""
        kT = self.kappa * T
        return self.theta + (self.v0 - self.theta) * (-np.expm1(-kT)) / kT

    def as_dict(self) -> dict:
        return asdict(self)
