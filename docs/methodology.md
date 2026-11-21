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

## Monte Carlo simulation (Phase 2)

Code: `src/heston_risk/mc/simulate.py`. Experiment: `scripts/run_phase2.py` -> `reports/phase2_results.md`.

### Why discretisation is hard for Heston

The variance follows a CIR process. Its exact one-step transition is a scaled non-central
chi-square distribution, which is expensive to sample (Broadie-Kaya exact simulation also needs
the conditional integrated variance via a Fourier inversion per step, which is far too slow for
risk work). A naive Euler step v' = v + kappa(theta - v)D + sigma sqrt(vD) Z can go negative,
and the fix chosen for the negative values (absorption, reflection, truncation) creates bias,
worst exactly when the Feller condition fails, which is the typical calibrated case.

### Baseline: full-truncation Euler

Lord, Koekkoek & van Dijk (2010) compare the fixes and find full truncation (use v+ = max(v,0)
in both drift and diffusion, but keep the untruncated v as the state) to be the least biased Euler
variant. It is first-order in dt (weakly) at best, and the constant can be large.

### Andersen (2008) quadratic-exponential (QE) scheme

**Variance step.** Instead of discretising the SDE, sample from a simple distribution whose first
two moments equal the exact conditional CIR moments m and s^2 (formulas in the module docstring).
Let psi = s^2/m^2:
* psi small (variance far from 0): v' = a (b + Z)^2, a squared shifted normal. This is the
  large-degrees-of-freedom shape of the non-central chi-square.
* psi large (variance near 0): a mixture of an atom at 0 with probability p and an exponential tail.
  This captures the real probability mass close to zero that the quadratic form cannot produce.
* The switch at psi_c = 1.5 (Andersen's choice) is valid because the quadratic form needs
  psi <= 2 and the exponential form needs psi >= 1. Both branches are non-negative by construction,
  so no truncation is ever needed.

**Price step.** Integrating the SDEs exactly gives
ln S' = ln S + (r-q)D + (rho/sigma)(v' - v - kappa theta D) + (kappa rho/sigma - 1/2) int v du
+ sqrt(1-rho^2) int sqrt(v) dW. The only approximation is int v du ~ D(gamma1 v + gamma2 v')
(central rule, gamma1 = gamma2 = 1/2), and the last term is then conditionally normal. Using v'
from the variance step keeps the price/variance correlation consistent with the sampled variance.

**Martingale correction.** The discretised scheme does not make the forward an exact martingale.
Because E[exp(A v') | v] is available in closed form for both branches, K0 is replaced path by path
so that E[S'|S, v] = S e^{(r-q)D} holds exactly. The tests show the drift without the correction
(about -1% per year-step for Andersen case I at v = 0.2) and none with it. If the closed form does
not exist (A too large) the plain K0 is used and counted; this never happened in the experiment.

### Estimators

* Independent samples are antithetic-pair averages, so the standard error is not understated.
* Optional control variate: the discounted terminal stock e^{-rT} S_T has known mean S0 e^{-qT}.
  The optimal coefficient is estimated from the same sample (a bias of order 1/N, negligible here).

### Bias measurement protocol

Truth is the Phase 1 analytic price (accurate to ~1e-10, far below MC error). For each case,
scheme and dt in {1, 1/2, ..., 1/32} years, 1,000,000 paths are simulated with a distinct fixed seed;
bias = MC - analytic is reported with its standard error and flagged significant if |bias| > 2 s.e.
A non-significant bias only means it is below the resolution of that path count.

### Findings (see `reports/phase2_results.md`)

* QE: no significant bias at any strike from dt = 1/4 onwards in cases B, I, II, III (from 1/8 in the
  mild case A, where a -0.018 bias at K=80 remains at dt = 1/4), including the three
  Andersen cases whose variance hits the exponential branch in 71% to 92% of draws.
* Euler: still significantly biased at dt = 1/32 in four of five cases; the bias is positive for
  the hard cases (truncation keeps variance artificially high) and decays roughly like dt.
* The martingale correction was always available (zero fallbacks).
* Antithetic variates help when the payoff is close to linear in the shocks (mild case, ~2x) but
  little in case I; the stock control variate gives ~2.5x there. Variance reduction does not
  touch discretisation bias, which is the dominant error for Euler.

## Calibration to SPY (Phase 3)

Code: `src/heston_risk/calibration/`. Experiment: `scripts/run_phase3.py` -> `reports/phase3_results.md`.
Data: `data/spy_2026-11-20/` (Yahoo end-of-day chain, 9,421 quotes, sha256 checked on load).

### Market inputs
* **Discount rate**: flat r from the 13-week T-bill (^IRX is a bank-discount yield d; P = 1 - d·91/360,
  r = -ln P / (91/365)). A free regression of C - P on K for (D, F) jointly was tried first and gave
  D > 1 on several expiries (too few, too noisy near-ATM pairs), so D is fixed and only F is estimated.
* **Forward per expiry**: F = median over near-ATM strikes of K + (C_mid - P_mid)/D. This absorbs dividends
  and the after-hours move of SPY (option quotes end ~15 minutes after the equity close).
* **Time**: ACT/365 from the last quote time to 16:00 New York on the expiry date.
* Each quote is priced with the Phase 1 pricer using S_eff = F e^{-rT}, q = 0, which reproduces F exactly.

### Quote selection
Out-of-the-money side only (puts K < F, calls K >= F), 7 days to 2.5 years, two-sided quotes,
relative spread <= 50%, mid >= $0.05, |ln(K/F)| <= max(0.5 sqrt(T), 0.1), IV invertible. OTM quotes are
the liquid ones and carry the least early-exercise value.

### American exercise
SPY options are American. The early-exercise premium of each quote is measured with a CRR tree
(American minus European on the same tree). Calls: zero. Puts: grows with maturity (median ~0.5 vol
points for 1-2 years) because r > q puts long-dated forward-OTM puts in the money against spot.
De-Americanisation: solve the tree volatility that reproduces the American mid, subtract the tree premium
at that volatility, re-invert the European IV. It is run as a sensitivity; the parameters move little.

### Objective and optimiser
Minimise sum_i ((C_model_i - C_mid_i)/vega_i)^2. Since dC = vega d(sigma), each term is approximately the
squared IV error, so this is an IV fit without inverting model prices inside the loop. Trust-region
least squares inside box bounds, 10 Latin-hypercube starts. Feller is not imposed; a Feller-constrained
fit (sigma = sqrt(2 kappa theta f), f in (0,1]) is reported for comparison.

### Validation
* Synthetic recovery test: prices generated with known parameters are calibrated back to them (tests).
* 80/20 split stratified by expiry; out-of-sample IV RMSE reported next to in-sample.
* Benchmark: one constant BS volatility.
* Multi-start agreement as a check on local minima.

### Results (2026-11-20 close, a -3.6% day)
Unconstrained: v0 = 0.133 (36% vol), kappa = 8.07, theta = 0.0556 (24% vol), sigma = 2.25, rho = -0.77,
Feller ratio 0.18. IV RMSE 0.96 (train) / 0.85 (test) vol points vs 6.8 / 6.6 for one BS vol;
median bid-ask spread is 0.31 vol points, so the fit is about 3 spreads wide on average.
All 10 starts reach the same optimum. The Feller-constrained fit is about 3x worse (2.8 / 2.5) and hits
rho = -0.999: without a large vol-of-vol the model cannot create the observed skew and curvature.
The very high kappa and sigma are the usual Heston compromise: one variance factor must produce both a
steep short-dated skew (needs large sigma) and a flatter long-dated term structure (needs fast reversion).

## Neural-network surrogate (Phase 4)

Code: `src/heston_risk/surrogate/`. Data: `scripts/make_phase4_data.py`. Training and evaluation:
`scripts/run_phase4.py` -> `reports/phase4_results.md`, trained weights `models/heston_iv_mlp.pt`.

### What the network learns, in plain terms
The analytic pricer is a function: give it (v0, kappa, theta, sigma, rho) and an option (strike,
maturity) and it returns a price, which we convert to an implied vol. It is exact but needs a numerical
integral for every maturity and parameter set. The MLP learns to imitate that same function.

* It is **not** a forecasting model. It never sees historical prices or dates; there is no noise in its
  labels. Every training label is an exact Heston IV computed by the Phase 1 pricer.
* Why an MLP suits it: the map is smooth and deterministic, with 7 inputs that interact non-linearly
  (rho tilts the smile, sigma bends it, both mostly at short maturities; kappa controls how fast v0 is
  forgotten as T grows). A stack of linear layers with smooth non-linear activations (SiLU) can
  approximate such a smooth function to high accuracy, and evaluating it is just a few small
  matrix multiplications, the same cost whatever the parameters or maturity.
* Why predict IV rather than price: IVs are all of similar size (5% to 60%) across strikes and
  maturities, whereas prices range from 1e-9 to 0.5 of the forward; a squared-error loss on prices
  would ignore cheap OTM options.
* Why only 7 inputs: with F = 1 and r = 0 the IV depends only on (params, ln(K/F), T); spot, rates and
  dividends cancel when price is converted back to IV (a test checks this at S = 693, r = 4%).

### Data design
Scrambled Sobol sampling of (params, T) slices over the box (T log-uniform), 10 strikes per slice;
the dataset is split into train / validation / test **by slice**, so strikes of one slice never sit on
both sides of the split (they are near-duplicates and would make the test error look better than it is).
The input scaling uses fixed box bounds, not statistics fitted on the data.

### Training
MSE on IV, Adam, learning rate halved after 3 epochs without validation improvement, early stopping.
The final model stopped at the epoch cap (150) while still improving slightly, so the reported accuracy is
not the limit of this architecture.

### Lessons recorded
* **Gradients through a float32 network.** The first surrogate calibration used finite-difference
  Jacobians (scipy default step ~1e-8). The network's float32 output is only accurate to ~1e-7, so those
  gradients were noise: the fit stopped early and its parameters had an IV RMSE of 1.58 vol points vs 0.94
  for the analytic calibration, with only 1 of 10 starts agreeing. Supplying the exact network gradient
  via autograd fixed it (10 of 10 starts agree, IV RMSE 0.92).
* The analytic calibration minimises vega-weighted *price* errors, a first-order proxy for IV errors;
  the surrogate fit minimises IV errors directly, which is why it can end with a marginally lower IV RMSE.
* No-arbitrage is not built in. Butterfly checks on 156,000 grid points found 6 violations (tiny
  wiggles in deep wings); calendar checks found none.

### Code reading guide
1. `surrogate/dataset.py`: `sample_slices` (what is sampled) and `label_slice` (how a label is made from
   the analytic pricer).
2. `surrogate/model.py`: `scale_inputs` (feature scaling), `IVNet` (the 4 x 128 network),
   `Surrogate.iv_and_param_jacobian` (autograd gradient used in calibration).
3. `surrogate/train.py`: `train_surrogate` (mini-batch loop, learning-rate schedule, early stopping).
4. `scripts/run_phase4.py`: section 6 shows how the surrogate replaces the pricer inside calibration.

## References

* Heston, S. (1993). A closed-form solution for options with stochastic volatility. RFS 6(2).
* Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.
* Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. Wilmott Magazine.
* Kahl, C., Jäckel, P. (2005). Not-so-complex logarithms in the Heston model. Wilmott Magazine.
* Fang, F., Oosterlee, C. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM J. Sci. Comput. 31(2).
* Lord, R., Koekkoek, R., van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. Quant. Finance 10(2).
* Broadie, M., Kaya, O. (2006). Exact simulation of stochastic volatility and other affine jump diffusion processes. Operations Research 54(2).
* Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. J. Comp. Finance 11(3).
