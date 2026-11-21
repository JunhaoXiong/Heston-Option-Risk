import numpy as np
from scipy.stats import norm

from heston_risk import HestonParams, heston_price
from heston_risk.risk.portfolio import MULTIPLIER, Book, Market, Option, book_value, delta_hedge_shares, greeks
from heston_risk.risk.var import full_pnl, historical_scenarios, run_methods, taylor_pnl, var_es

S, R = 100.0, 0.03
P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
M = Market(S=S, v=P.v0, r=R, params=P)


def bs_greeks(S, K, T, r, q, vol, kind):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    delta = np.exp(-q * T) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)       # per unit of vol
    return delta, gamma, vega


def test_greeks_reduce_to_black_scholes_without_vol_of_vol():
    """sigma -> 0 and v0 = theta: variance is constant, Heston = Black-Scholes with vol sqrt(v0).
    vega_v = dV/dv = BS vega / (2 sqrt v)."""
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=0.0)
    m = Market(S, 0.04, R, p)
    for kind, K in (("call", 105.0), ("put", 95.0)):
        o = Option(kind, K, 0.5, 0.01, 1)
        g = greeks(Book((o,)), m)
        d, gm, vg = bs_greeks(S, K, 0.5, R, 0.01, 0.2, kind)
        assert abs(g["delta"] / MULTIPLIER - d) < 2e-4
        assert abs(g["gamma"] / MULTIPLIER - gm) < 2e-4
        # with v0 = theta, kappa only matters through the bump; dV/dv is close to BS vega/(2 sqrt v)
        # times the average-variance sensitivity (1 - e^{-kappa T})/(kappa T)
        w = (1 - np.exp(-1.5 * 0.5)) / (1.5 * 0.5)
        assert abs(g["vega_v"] / MULTIPLIER - vg / (2 * 0.2) * w) / (vg / 0.4 * w) < 1e-3


def test_call_put_parity_of_greeks():
    T, K, q = 0.25, 100.0, 0.01
    c = greeks(Book((Option("call", K, T, q, 1),)), M)
    p = greeks(Book((Option("put", K, T, q, 1),)), M)
    assert abs((c["delta"] - p["delta"]) / MULTIPLIER - np.exp(-q * T)) < 1e-5
    for k in ("gamma", "vega_v", "vanna", "volga"):
        assert abs(c[k] - p[k]) <= 1e-3 * max(1.0, abs(c[k]))


def test_delta_hedge_neutralises_delta():
    legs = (Option("call", 100.0, 0.1, 0.0, -10), Option("put", 100.0, 0.1, 0.0, -10),
            Option("put", 90.0, 0.25, 0.0, 10))
    b = Book(legs, delta_hedge_shares(Book(legs), M))
    assert abs(greeks(b, M)["delta"]) < 1e-6


def test_second_order_taylor_converges_to_full_revaluation():
    """For shrinking (dS, dv) the second-order Taylor error falls like the third power of the move."""
    b = Book((Option("call", 100.0, 0.25, 0.0, 1), Option("put", 95.0, 0.5, 0.0, -2)))
    g = greeks(b, M)
    errs = []
    for eps in (0.02, 0.01):
        dS, dv = np.array([eps * S]), np.array([eps * 0.5 * M.v])
        full = full_pnl(b, M, S + dS, M.v + dv, 0.0)
        errs.append(abs(taylor_pnl(g, dS, dv, 0.0, "dgv") - full)[0])
    assert errs[1] < errs[0] / 4  # at least second-order better than delta-gamma would be


def test_zero_scenario_gives_theta_only():
    b = Book((Option("call", 100.0, 0.25, 0.0, 1),))
    pnl, _, g, _ = run_methods(b, M, np.zeros(1), np.ones(1), 1 / 252)
    for k in ("delta", "delta_gamma", "dgv"):
        assert np.isclose(pnl[k][0], g["theta"] / 252)
    assert abs(pnl["full"][0] - g["theta"] / 252) < 0.02 * abs(g["theta"] / 252)


def test_option_value_matches_pricer_and_expiry_payoff():
    o = Option("put", 97.0, 0.3, 0.01, -3)
    assert np.isclose(book_value(Book((o,), 5.0), M),
                      -3 * MULTIPLIER * float(heston_price(S, 97.0, 0.3, R, 0.01, P, "put")) + 5 * S)
    assert np.isclose(book_value(Book((o,)), M, S=90.0, dt=0.3), -3 * MULTIPLIER * 7.0)


def test_var_es_on_known_sample():
    pnl = -np.arange(1, 1001, dtype=float)      # losses 1..1000
    v, e = var_es(pnl, 0.99, 0.975)
    assert np.isclose(v, np.quantile(np.arange(1, 1001), 0.99))
    assert np.isclose(e, np.arange(1, 1001)[np.arange(1, 1001) >= np.quantile(np.arange(1, 1001), 0.975)].mean())


def test_historical_scenarios_are_forward_looking_pairs():
    spy = np.array([100.0, 101.0, 99.0, 102.0])
    vix = np.array([20.0, 22.0, 18.0, 18.0])
    x, r = historical_scenarios(spy, vix, 2)
    assert np.allclose(x, np.log([99 / 100, 102 / 101]))
    assert np.allclose(r, [18 / 20, 18 / 22])
