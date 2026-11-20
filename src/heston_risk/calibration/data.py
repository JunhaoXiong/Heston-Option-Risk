"""SPY option-chain snapshot -> clean out-of-the-money quote set with implied forwards and IVs.

Pipeline (each step is a separate function so it can be tested and reported):
  load_snapshot      raw Yahoo chain (as stored under data/), integrity check against meta.json sha256
  rate_from_irx      flat continuously-compounded rate from the 13-week T-bill discount yield
  year_fractions     ACT/365 from the quote time to expiry at 16:00 New York time
  implied_forwards   per expiry, median of K + (C_mid - P_mid)/D over near-the-money strikes
  build_quotes       OTM selection relative to F, liquidity / moneyness filters, mid prices, IVs, vegas

Conventions: the model is priced on the forward, so a quote is represented by an
effective spot S_eff = F e^{-rT} with q = 0, which reproduces the implied forward F
exactly with the Phase 1 pricer interface (S, K, T, r, q).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from heston_risk.bs import bs_vega, implied_vol

NY_CLOSE_UTC_HOUR_EST = 21  # 16:00 New York in November (EST = UTC-5)


def load_snapshot(folder: str | Path, verify: bool = True):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    if verify:
        for name, digest in meta["sha256"].items():
            got = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if got != digest:
                raise ValueError(f"sha256 mismatch for {name}")
    raw = pd.read_csv(folder / "spy_options_raw.csv")
    raw["lastTradeDate"] = pd.to_datetime(raw["lastTradeDate"], utc=True)
    irx = pd.read_csv(folder / "irx_10d.csv")
    hist = pd.read_csv(folder / "spy_history_10d.csv")
    return raw, irx, hist, meta


def rate_from_irx(discount_yield_pct: float, days: int = 91) -> float:
    """^IRX is a bank-discount yield: P = 1 - d * days/360. Return -ln(P)/(days/365)."""
    P = 1.0 - discount_yield_pct / 100.0 * days / 360.0
    return float(-np.log(P) / (days / 365.0))


def year_fractions(expiries: pd.Series, asof: pd.Timestamp) -> pd.Series:
    exp_ts = pd.to_datetime(expiries).dt.tz_localize("UTC") + pd.Timedelta(hours=NY_CLOSE_UTC_HOUR_EST)
    return (exp_ts - asof).dt.total_seconds() / (365.0 * 86400.0)


@dataclass
class ForwardFit:
    expiry: str
    T: float
    F: float
    n_strikes: int
    iqr: float


def implied_forwards(raw: pd.DataFrame, r: float, asof: pd.Timestamp, spot_ref: float,
                     band: float = 0.04, min_strikes: int = 3) -> pd.DataFrame:
    """Per-expiry forward from put-call parity C - P = D (F - K) with D = e^{-rT} fixed.

    Uses strikes with two-sided quotes on both the call and the put and |ln(K/spot_ref)| < band;
    the median of the per-strike estimates is robust to a few bad quotes.
    """
    out = []
    for e, d in raw.groupby("expiry"):
        T = float(year_fractions(pd.Series([e]), asof).iloc[0])
        if T <= 0:
            continue
        c = d[d.kind == "call"].set_index("strike")
        p = d[d.kind == "put"].set_index("strike")
        ks = c.index.intersection(p.index)
        ks = ks[np.abs(np.log(ks / spot_ref)) < band]
        ok = [k for k in ks if c.at[k, "bid"] > 0 and c.at[k, "ask"] > 0 and p.at[k, "bid"] > 0 and p.at[k, "ask"] > 0]
        if len(ok) < min_strikes:
            continue
        D = np.exp(-r * T)
        est = np.array([k + (0.5 * (c.at[k, "bid"] + c.at[k, "ask"]) - 0.5 * (p.at[k, "bid"] + p.at[k, "ask"])) / D
                        for k in ok])
        q75, q25 = np.percentile(est, [75, 25])
        out.append(ForwardFit(e, T, float(np.median(est)), len(ok), float(q75 - q25)))
    return pd.DataFrame([f.__dict__ for f in out])


@dataclass(frozen=True)
class QuoteFilter:
    min_days: float = 7.0
    max_years: float = 2.5
    max_rel_spread: float = 0.5        # (ask - bid) / mid
    moneyness_width: float = 0.5       # keep |ln(K/F)| <= width * sqrt(T) ...
    moneyness_floor: float = 0.1       # ... but never narrower than this
    min_price: float = 0.05            # mid price in $


def build_quotes(raw: pd.DataFrame, fwd: pd.DataFrame, r: float, flt: QuoteFilter = QuoteFilter()):
    """Return (quotes, funnel): OTM quotes with mid, IV and vega, plus a filter funnel table."""
    funnel = [("raw quotes", len(raw))]
    df = raw.merge(fwd[["expiry", "T", "F"]], on="expiry", how="inner")
    funnel.append(("expiry has an implied forward", len(df)))
    df = df[(df["T"] >= flt.min_days / 365.0) & (df["T"] <= flt.max_years)]
    funnel.append((f"{flt.min_days:g} days <= T <= {flt.max_years:g} y", len(df)))
    otm = ((df.kind == "put") & (df.strike < df.F)) | ((df.kind == "call") & (df.strike >= df.F))
    df = df[otm]
    funnel.append(("out-of-the-money side only", len(df)))
    df = df[(df.bid > 0) & (df.ask > df.bid)]
    funnel.append(("two-sided quote (bid > 0, ask > bid)", len(df)))
    df = df.assign(mid=0.5 * (df.bid + df.ask))
    df = df[(df.ask - df.bid) / df.mid <= flt.max_rel_spread]
    funnel.append((f"relative spread <= {flt.max_rel_spread:.0%}", len(df)))
    df = df[df.mid >= flt.min_price]
    funnel.append((f"mid >= ${flt.min_price}", len(df)))
    k = np.log(df.strike / df.F)
    width = np.maximum(flt.moneyness_width * np.sqrt(df["T"]), flt.moneyness_floor)
    df = df[np.abs(k) <= width]
    funnel.append(("|ln(K/F)| <= max(0.5 sqrt(T), 0.1)", len(df)))
    df = df.copy()
    df["k"] = np.log(df.strike / df.F)
    df["S_eff"] = df.F * np.exp(-r * df["T"])
    df["iv"] = implied_vol(df.mid.values, df.S_eff.values, df.strike.values, df["T"].values, r, 0.0,
                           df.kind.values)
    df = df[np.isfinite(df.iv)]
    funnel.append(("implied vol solvable (inside no-arbitrage bounds)", len(df)))
    df["vega"] = bs_vega(df.S_eff.values, df.strike.values, df["T"].values, r, 0.0, df.iv.values)
    df["iv_bid"] = implied_vol(df.bid.values, df.S_eff.values, df.strike.values, df["T"].values, r, 0.0, df.kind.values)
    df["iv_ask"] = implied_vol(df.ask.values, df.S_eff.values, df.strike.values, df["T"].values, r, 0.0, df.kind.values)
    cols = ["expiry", "T", "F", "S_eff", "kind", "strike", "k", "bid", "ask", "mid", "iv", "iv_bid", "iv_ask",
            "vega", "volume", "openInterest", "impliedVolatility"]
    return df[cols].sort_values(["T", "strike"]).reset_index(drop=True), pd.DataFrame(funnel, columns=["step", "rows"])
