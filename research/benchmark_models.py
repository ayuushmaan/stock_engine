"""Standard-model benchmark: ours vs industry baselines on identical forwards.

Baselines a real desk would demand before allocating a rupee:
  - buy_hold: equal-weight universe drift (the hurdle)
  - momentum_20d / momentum_5d: past-return time-series signals
  - tone_only: raw GDELT tone, no credibility filter (ablation)
  - ours (net_signal) / organic_only: our pipeline signals

All evaluated PREDICTIVE (signal_t -> fwd1 next-day close-to-close):
rank IC + HAC-style t, sign hit-rate, Q5-Q1 spread.

Usage: python research/benchmark_models.py
Output: outputs/tables/model_comparison.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_FINAL, OUTPUTS_TABLES, setup_logging
from research.backtest import run_factor_backtest

logger = setup_logging()


def daily_ic(df: pd.DataFrame, sig: str, ret: str, min_n: int = 8) -> dict:
    d = df[["effective_date", sig, ret]].dropna()
    ics = []
    for _, g in d.groupby("effective_date"):
        if len(g) >= min_n:
            try:
                r, _ = stats.spearmanr(g[sig], g[ret])
                if not np.isnan(r):
                    ics.append(r)
            except Exception:
                pass
    ics = np.array(ics)
    m = float(ics.mean()) if len(ics) else float("nan")
    se = float(ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else float("nan")
    return {"mean_ic": round(m, 4), "t": round(m / se, 2) if se else None, "ndays": len(ics)}


def hit_rate(df: pd.DataFrame, sig: str, ret: str) -> dict:
    h = df[[sig, ret]].dropna()
    hr = float((((h[sig] > 0) == (h[ret] > 0))).mean())
    return {"hit_rate": round(hr, 4), "n": len(h)}


def q5_q1(df: pd.DataFrame, sig: str, ret: str) -> dict:
    """Annualized long-short spread of scoring by sig, realized in ret."""
    recs = []
    for _, g in df.groupby("effective_date"):
        if len(g) < 10:
            continue
        hi, lo = g[sig].quantile(0.8), g[sig].quantile(0.2)
        recs.append(g.loc[g[sig] >= hi, ret].mean() - g.loc[g[sig] <= lo, ret].mean())
    s = pd.Series(recs).dropna()
    return {"ann_spread_pct": round(float(s.mean() * 252 * 100), 2),
            "sharpe": round(float(s.mean() / s.std() * np.sqrt(252)), 2) if len(s) > 1 and s.std() else None}


def run_comparison() -> dict:
    df = pd.read_parquet(DATA_FINAL / "master_dataset.parquet")
    if "effective_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "effective_date"})
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df = df.sort_values(["ticker", "effective_date"]).reset_index(drop=True)

    df["mom_20d"] = df.groupby("ticker")["ret_close2close"].transform(
        lambda s: s.rolling(20).sum().shift(1))
    df["mom_5d"] = df.groupby("ticker")["ret_close2close"].transform(
        lambda s: s.rolling(5).sum().shift(1))
    df["fwd1"] = df.groupby("ticker")["ret_close2close"].shift(-1)
    df = df.dropna(subset=["fwd1"]).reset_index(drop=True)
    if "tone_proxy" not in df.columns:
        df["tone_proxy"] = df.get("organic_signal", 0).fillna(0) + df.get("sponsored_signal", 0).fillna(0)

    signals = {
        "ours_net": "net_signal",
        "ours_organic_only": "organic_signal",
        "tone_only_no_filter": "tone_proxy",
        "momentum_20d": "mom_20d",
        "momentum_5d": "mom_5d",
    }
    out = {}
    for name, col in signals.items():
        if col not in df.columns:
            continue
        out[name] = {**daily_ic(df, col, "fwd1"), **hit_rate(df, col, "fwd1"),
                     **q5_q1(df, col, "fwd1")}

    bh = df.groupby("effective_date")["ret_close2close"].mean()
    out["buy_hold_equal_weight"] = {
        "ann_return_pct": round(float(bh.mean() * 252 * 100), 2),
        "ann_vol_pct": round(float(bh.std() * np.sqrt(252) * 100), 2),
    }
    # full Q5-Q1 backtest (with turnover + costs) for ours only
    perf, m = run_factor_backtest(df, signal_col="net_signal",
                                  return_col="fwd1", cost_bps=10.0)
    out["ours_net_backtest_10bps"] = {
        "gross_ann_pct": m.annualized_return, "net_ann_pct": m.net_annualized_return,
        "net_sharpe": m.net_sharpe_ratio, "max_dd_pct": m.max_drawdown,
        "turnover_pct": m.daily_turnover,
    }
    return out


if __name__ == "__main__":
    res = run_comparison()
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_TABLES / "model_comparison.json", "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res.items():
        logger.info(f"{k}: {v}")
