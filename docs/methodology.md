# Methodology

This document records the mathematics, the numerical choices and the reasons for them.
It grows one section per phase.

## Notation and model (Phase 1)

Under the risk-neutral measure with constant rate r and dividend yield q,

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t,      d<W^S, W^v>_t = rho dt

| Symbol | Meaning | Constraint |
|---|---|---|
| v0 | initial variance | >= 0 |
| kappa | mean-reversion speed of variance | > 0 |
| theta | long-run variance | >= 0 |
| sigma | volatility of variance ("vol-of-vol") | > 0 |
| rho | correlation of spot and variance shocks | (-1, 1) |

Feller condition 2 kappa theta >= sigma^2 keeps v_t strictly positive. It is reported but
**not** imposed: calibrated equity parameters often violate it (the Fang-Oosterlee benchmark
has ratio 0.38), and the Fourier pricer is valid either way. It matters for Monte Carlo (Phase 2).

Forward F = S e^{(r-q)T}; all pricing is done on the forward, so r and q only enter through F
and the discount factor e^{-rT}.

## Characteristic function

Let X_T = ln(S_T / F). Heston's affine structure gives phi(u) = E[e^{iuX_T}] = exp(C(u,T) + D(u,T) v0)
with C, D solving Riccati ODEs. Writing b = kappa - rho sigma i u and
d = sqrt(b^2 + sigma^2 (iu + u^2)) (principal root, Re d >= 0), there are two algebraically
equivalent closed forms:

| | g | exponential | log term |
|---|---|---|---|
| Heston (1993) | g1 = (b + d)/(b - d) | e^{+dT} | ln((1 - g1 e^{dT}) / (1 - g1)) |
| Albrecher et al. (2007), "little trap" | g2 = (b - d)/(b + d) = 1/g1 | e^{-dT} | ln((1 - g2 e^{-dT}) / (1 - g2)) |

    little trap:  C = kappa theta / sigma^2 [ (b - d) T - 2 ln((1 - g2 e^{-dT}) / (1 - g2)) ]
                  D = (b - d) / sigma^2 * (1 - e^{-dT}) / (1 - g2 e^{-dT})

### Why the little-trap form

This is a purely **numerical** choice. Both forms are exact solutions of the same Riccati
equations (equivalently, the same Heston pricing PDE); the little trap does not make the PDE
easier to solve and does not change the model.

The issue is the complex logarithm. Any library `log` returns the principal branch, with a cut
along the negative real axis. phi must be a continuous function of u along the integration path
u in [0, inf). In the original form the argument (1 - g1 e^{dT})/(1 - g1) rotates around the
origin as u grows (faster for long T and large sigma), so each time it crosses the negative real
axis the principal log jumps by 2 pi i. Because C is multiplied by kappa theta / sigma^2, which is
generally not an integer, exp(C) jumps as well: phi becomes discontinuous and the price is wrong.
The "spiral" picture is exactly this winding of the log argument in the complex plane.

Albrecher, Mayer, Schoutens & Tistaert (2007) prove that in the g2 / e^{-dT} form the log argument
does not cross the negative real axis along the integration path (under mild conditions that hold
for realistic parameters), so the principal branch *is* the continuous branch. We also check this
empirically: the little-trap phi is continuous on fine u-grids in the tests, and its prices agree
with an independent formula across 400 random parameter sets.
That avoids the alternative fix of tracking the winding number (Kahl & Jaeckel 2005). A side benefit:
e^{-dT} decays, so nothing overflows, whereas e^{+dT} overflows for large u.

Evidence in this repo (`scripts/run_phase1.py`, section 6; `tests/test_heston_cf.py`): with
v0 = theta = 0.04, kappa = 2, sigma = 0.6, rho = -0.8 the original form already misprices the
1-year ATM call by about 0.09 and the 5-year ATM call by more than 100%, while the little-trap form
agrees with the independent Gil-Pelaez formula to ~1e-10.

## Pricing formula

Primary: Lewis (2001) single integral, a contour shifted to Im u = -1/2:

    C = e^{-rT} [ F - sqrt(F K) / pi * int_0^inf Re( e^{iuk} phi(u - i/2) ) / (u^2 + 1/4) du ],   k = ln(F/K)

Puts by put-call parity P = C - e^{-rT}(F - K). Reasons for preferring Lewis over Heston's
two-probability (Gil-Pelaez) formula:
* one integral instead of two;
* the integrand is bounded at u = 0 (no 1/u singularity) and decays like |phi| / u^2;
* the CF depends only on T, so for a fixed maturity one CF evaluation serves every strike
  (vectorised as a matrix product). This matters for calibration (Phase 3).

### Quadrature

Composite Gauss-Legendre on [0, U] with 24 nodes per panel; panel edges are graded near the
origin (0, 1/4, 1/2, 1, ..., 16) because the kernel 1/(u^2 + 1/4) varies on an O(1) scale there,
then uniform panels of width 20. U is the point beyond which the strike-independent envelope
|phi(u - i/2)| / (u^2 + 1/4) is below 1e-14. Short maturities have a slowly decaying CF and get
more panels automatically (about 600 nodes at 1 week, 200 at 5 years).

A single Gauss-Legendre rule on [0, U] was tried first and rejected: it under-resolves the
region near u = 0 and gave errors up to 2e-4 on short-dated options. This was caught by the
cross-check against the independent formula below.

### Validation (all automated in `tests/`)

1. **Literature benchmark**: Fang & Oosterlee (2008) case, published 5.785155450. We obtain
   5.7851554344 with three methods and 5.78515543437619 with 30-digit mpmath; the 1.6e-8 gap to
   the published figure is in its last digits.
2. **Independent formula**: Gil-Pelaez two-probability formula with adaptive quadrature, puts priced
   directly (not via parity): max absolute difference 1.1e-10 over 400 random cases from 1 week to 5y.
3. **Black-Scholes limit**: as sigma -> 0 the variance becomes deterministic and the price must tend
   to BS with variance equal to the time-average of E[v_t]; error is second order in sigma when
   rho = 0 and first order otherwise.
4. **Monte Carlo**: full-truncation Euler, 200k antithetic paths, all |z| < 2.
5. **No-arbitrage shape**: bounds, monotone decreasing and convex in strike; negative rho gives
   negative skew.

### Scope and assumptions of Phase 1

* European options only; constant r and q (for SPY, Phase 3 will imply the forward per expiry
  from put-call parity rather than assume r and q).
* Prices are per unit of underlying, no contract multiplier.
* No greeks yet (Phase 5).

## References

* Heston, S. (1993). A closed-form solution for options with stochastic volatility. RFS 6(2).
* Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.
* Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. Wilmott Magazine.
* Kahl, C., Jäckel, P. (2005). Not-so-complex logarithms in the Heston model. Wilmott Magazine.
* Fang, F., Oosterlee, C. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM J. Sci. Comput. 31(2).
* Lord, R., Koekkoek, R., van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. Quant. Finance 10(2).
* Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. J. Comp. Finance 11(3).
