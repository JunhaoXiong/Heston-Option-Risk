# Heston-Option-Risk

Heston stochastic-volatility option pricing, calibration to real SPY options, a neural-network
surrogate pricer, and nonlinear (options) portfolio risk. Companion to
[Market-Risk-Engine](https://github.com/JunhaoXiong/Market-Risk-Engine), which covers linear
ETF-portfolio VaR/ES and backtesting.

Built phase by phase, one pull request per phase, each with tests and a generated results report.
No number in the reports is typed by hand: every table is produced by a script in `scripts/`.

| Phase | Content | Status |
|---|---|---|
| 1 | Analytic Heston pricer (characteristic function, Lewis formula), validation | done |
| 2 | Monte Carlo with Andersen (2008) QE scheme vs analytic; bias/variance study | done |
| 3 | Calibration to a real SPY implied-volatility surface | done |
| 4 | Neural-network surrogate pricer trained on analytic prices; honest speed/error report | planned |
| 5 | Greeks; delta-gamma vs full-revaluation VaR for an options portfolio | planned |

## Phase 1 highlights

Full report: [`reports/phase1_results.md`](reports/phase1_results.md). Method and reasoning:
[`docs/methodology.md`](docs/methodology.md).

* Fang & Oosterlee (2008) benchmark: 5.7851554344 (30-digit mpmath: 5.78515543437619).
* Primary pricer vs an independent formula (Gil-Pelaez), 400 random cases from 1 week to 5 years:
  max absolute difference 1.1e-10.
* About 1 ms for 50 strikes at one maturity, roughly 100x faster than adaptive quadrature.
* Uses the "little Heston trap" characteristic function. With the original 1993 form and a standard
  complex log, the 5-year ATM call in our test case comes out at 36.0 instead of 15.9.

## Phase 2 highlights

Full report: [`reports/phase2_results.md`](reports/phase2_results.md).

* Andersen QE with martingale correction against full-truncation Euler, 5 parameter sets (incl. the three
  hard Andersen 2008 cases with Feller ratios 0.03 to 0.18), dt from 1 to 1/32 years, 1,000,000 paths per point,
  truth = Phase 1 analytic price.
* QE has no statistically significant bias from dt = 1/4 on in the four Feller-violating cases (from 1/8 in the mild case); full-truncation Euler is still
  significantly biased at dt = 1/32 in four of five cases (e.g. Andersen case I, ATM: +0.235 on a 13.08 price).
* At dt = 1/8 the worst QE bias is 0.019 vs 1.16 for Euler. Euler's bias shrinks roughly linearly in dt
  (fitted slopes 0.84 to 1.24), i.e. first order but with a large constant.
* Variance reduction: antithetic sampling halves variance in the mild case but does little in the hard
  case; the stock control variate adds a 2.5x reduction there.

## Phase 3 highlights

Full report: [`reports/phase3_results.md`](reports/phase3_results.md). Data snapshot in `data/spy_2026-11-20/`.

* 2,962 out-of-the-money SPY quotes (23 expiries, 10 days to 2.2 years) from the 2026-11-20 close, forwards
  implied per expiry from put-call parity, IVs inverted in-house.
* Calibrated Heston: v0 = 0.133, kappa = 8.07, theta = 0.056, sigma = 2.25, rho = -0.77 (Feller ratio 0.18).
* IV RMSE 0.96 vol points in-sample and 0.85 on a 20% hold-out, vs 6.8 / 6.6 for a single Black-Scholes vol.
  All 10 random starts converge to the same optimum.
* Imposing Feller triples the error and drives rho to -0.999.
* Early exercise matters for long-dated puts (median 0.5 vol points beyond one year); a de-Americanised
  re-fit moves the parameters only slightly.

## Why this project exists (rebuild of earlier code)

This repository replaces an earlier notebook of mine. Its problems, and how each phase fixes them:

| Earlier code | Problem | Fix here |
|---|---|---|
| Part 1 NN | Training labels were random grid points; no Heston price was ever computed | Labels are analytic Heston prices validated to ~1e-10 (Phases 1 and 4) |
| Part 2 MC | Each training sample used **one** simulated path as its "price" (pure noise, no averaging) | Proper MC estimator with standard errors and a bias study (Phase 2) |
| Part 2 MC | Paths simulated to T = 1 but discounted with a random T_sample | Consistent maturity throughout; tested |
| Part 2 MC | Euler with `max(v, 0)` and no check against an exact price | Andersen QE vs analytic, convergence tables (Phase 2) |
| Both | No calibration, no real data, no validation | Calibrated to a real SPY surface with out-of-sample check (Phase 3), tests in every phase |

## Layout

```
src/heston_risk/
  bs.py                 Black-Scholes price, vega, implied vol
  heston/params.py      parameter container, Feller ratio
  heston/charfunc.py    characteristic function (little trap; 1993 form kept for the demo)
  heston/pricer.py      Lewis + composite Gauss-Legendre; quad and Gil-Pelaez references
  mc/euler.py           full-truncation Euler (Phase 1 sanity check)
  calibration/          snapshot loading, implied forwards, quote filters, calibration, American-premium check
  mc/simulate.py        Andersen QE + full-truncation Euler, path grid output, MC pricer with s.e. and control variate
tests/                  pytest suite
scripts/run_phase1.py   regenerates reports/phase1_results.md and figures
scripts/run_phase2.py   QE vs Euler bias study (about 4 minutes; --quick for a smoke test)
scripts/run_phase3.py   SPY calibration report (about 3 minutes)
data/spy_2026-11-20/    raw option-chain snapshot + sha256 metadata
docs/methodology.md     formulas, numerical choices, references
```

## Usage

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_phase1.py
```

```python
from heston_risk import HestonParams, heston_price, implied_vol
p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
heston_price(S=100, K=[90, 100, 110], T=0.5, r=0.03, q=0.01, params=p, kind="call")
```
