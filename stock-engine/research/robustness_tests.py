"""Robustness Tests — Placebo, Permutation, Global FDR, and Holding-Period Sweeps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    DATA_FINAL,
    DATA_PROCESSED,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    setup_logging,
)

logger = setup_logging()


def placebo_shuffled_dates(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str = "ret_fwd_1d",
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Placebo test: shuffle dates relative to signals.
    Destroys temporal alignment while preserving marginal distributions.
    """
    np.random.seed(seed)
    date_col = "effective_date" if "effective_date" in df.columns else "date"
    
    # Original IC
    daily_ics_orig = df.groupby(date_col, group_keys=False).apply(
        lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
    ).dropna()
    orig_mean_ic = float(daily_ics_orig.mean())
    
    # Permutation: shuffle date labels within signal-return pairs
    # Better approach: randomly reassign returns to different dates
    placebo_means = []
    unique_dates = df[date_col].unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    
    # Pre-compute daily IC function
    def compute_daily_ics(df_local):
        return df_local.groupby(date_col, group_keys=False).apply(
            lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
        ).dropna()
    
    for _ in range(n_permutations):
        # Shuffle the return column globally across dates
        df_perm = df.copy()
        df_perm[return_col] = np.random.permutation(df_perm[return_col].values)
        
        daily_ics = compute_daily_ics(df_perm)
        if len(daily_ics) > 0:
            placebo_means.append(float(daily_ics.mean()))
    
    placebo_means = np.array(placebo_means)
    
    # p-value: proportion of placebo >= observed (one-tailed)
    p_value = float(np.mean(placebo_means >= orig_mean_ic))
    
    return {
        "original_mean_ic": round(orig_mean_ic, 6),
        "placebo_mean": round(float(placebo_means.mean()), 6),
        "placebo_std": round(float(placebo_means.std()), 6),
        "p_value": round(p_value, 6),
        "p_value_twotailed": round(2 * min(p_value, 1 - p_value), 6),
        "n_permutations": n_permutations,
        "ci_95": [round(float(np.percentile(placebo_means, 2.5)), 6), round(float(np.percentile(placebo_means, 97.5)), 6)],
    }


def placebo_shuffled_tickers(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str = "ret_fwd_1d",
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Placebo test: shuffle tickers within each date.
    Destroys cross-sectional alignment while preserving daily structure.
    """
    np.random.seed(seed)
    date_col = "effective_date" if "effective_date" in df.columns else "date"
    
    daily_ics_orig = df.groupby(date_col).apply(
        lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
    ).dropna()
    orig_mean_ic = float(daily_ics_orig.mean())
    
    placebo_means = []
    
    for _ in range(n_permutations):
        df_shuffled = df.copy()
        # Shuffle signal within each date
        df_shuffled[signal_col] = df_shuffled.groupby(date_col)[signal_col].transform(
            lambda x: np.random.permutation(x.values) if len(x) > 1 else x
        )
        
        daily_ics = df_shuffled.groupby(date_col).apply(
            lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
        ).dropna()
        
        if len(daily_ics) > 0:
            placebo_means.append(float(daily_ics.mean()))
    
    placebo_means = np.array(placebo_means)
    p_value = float(np.mean(placebo_means >= orig_mean_ic))
    
    return {
        "original_mean_ic": round(orig_mean_ic, 6),
        "placebo_mean": round(float(placebo_means.mean()), 6),
        "placebo_std": round(float(placebo_means.std()), 6),
        "p_value": round(p_value, 6),
        "p_value_twotailed": round(2 * min(p_value, 1 - p_value), 6),
        "n_permutations": n_permutations,
        "ci_95": [round(float(np.percentile(placebo_means, 2.5)), 6), round(float(np.percentile(placebo_means, 97.5)), 6)],
    }


def permutation_test_ic(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str = "ret_fwd_1d",
    n_permutations: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Permutation test for IC significance.
    Under H0: signal and return are independent.
    """
    np.random.seed(seed)
    date_col = "effective_date" if "effective_date" in df.columns else "date"
    
    daily_ics_orig = df.groupby(date_col).apply(
        lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
    ).dropna()
    orig_mean_ic = float(daily_ics_orig.mean())
    
    perm_means = []
    
    for _ in range(n_permutations):
        # Permute return column globally (breaks all structure)
        perm_returns = np.random.permutation(df[return_col].values)
        df_perm = df.copy()
        df_perm[return_col] = perm_returns
        
        daily_ics = df_perm.groupby(date_col).apply(
            lambda g: stats.spearmanr(g[signal_col].fillna(0), g[return_col]).statistic if len(g) >= 5 else np.nan
        ).dropna()
        
        if len(daily_ics) > 0:
            perm_means.append(float(daily_ics.mean()))
    
    perm_means = np.array(perm_means)
    p_value = float(np.mean(np.abs(perm_means) >= np.abs(orig_mean_ic)))
    
    return {
        "original_mean_ic": round(orig_mean_ic, 6),
        "perm_mean": round(float(perm_means.mean()), 6),
        "perm_std": round(float(perm_means.std()), 6),
        "p_value_twotailed": round(p_value, 6),
        "n_permutations": n_permutations,
        "ci_95": [round(float(np.percentile(perm_means, 2.5)), 6), round(float(np.percentile(perm_means, 97.5)), 6)],
    }


def global_benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR across ALL tests (global)."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    adjusted_p = np.zeros(n)
    for i in range(n):
        rank = i + 1
        adjusted_p[i] = min(1.0, sorted_p[i] * n / rank)
    
    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    res = np.zeros(n)
    res[sorted_indices] = adjusted_p
    return res.tolist()


def holding_period_sweep(
    df: pd.DataFrame,
    signal_col: str,
    max_horizon: int = 20,
    cost_bps: float = 10.0,
) -> Dict[str, Any]:
    """
    Backtest sweep across holding periods 1..max_horizon days.
    Reports gross/net returns, Sharpe, turnover for each horizon.
    """
    date_col = "effective_date" if "effective_date" in df.columns else "date"
    df = df.sort_values([date_col, "ticker"]).copy()
    
    results = {}
    
    for h in range(1, max_horizon + 1):
        fwd_col = f"ret_fwd_{h}d"
        if fwd_col not in df.columns:
            logger.warning(f"Missing {fwd_col}, stopping sweep at h={h-1}")
            break
        
        daily_records = []
        unique_dates = df[date_col].unique()
        prev_longs = set()
        prev_shorts = set()
        
        for d in unique_dates:
            day_slice = df[df[date_col] == d]
            if len(day_slice) < 10:
                continue
            
            q_high = day_slice[signal_col].quantile(0.80)
            q_low = day_slice[signal_col].quantile(0.20)
            
            long_basket = day_slice[day_slice[signal_col] >= q_high]
            short_basket = day_slice[day_slice[signal_col] <= q_low]
            
            ret_long = float(long_basket[fwd_col].mean()) if len(long_basket) > 0 else 0.0
            ret_short = float(short_basket[fwd_col].mean()) if len(short_basket) > 0 else 0.0
            gross_return = ret_long - ret_short
            
            curr_longs = set(long_basket["ticker"]) if "ticker" in long_basket.columns else set()
            curr_shorts = set(short_basket["ticker"]) if "ticker" in short_basket.columns else set()
            
            if prev_longs and curr_longs:
                long_turnover = len(curr_longs - prev_longs) / max(1, len(curr_longs))
                short_turnover = len(curr_shorts - prev_shorts) / max(1, len(curr_shorts))
                turnover = (long_turnover + short_turnover) / 2.0
            else:
                turnover = 1.0
            
            prev_longs = curr_longs
            prev_shorts = curr_shorts
            
            tx_cost = 2.0 * turnover * (cost_bps / 10000.0)
            net_return = gross_return - tx_cost
            
            daily_records.append({
                "date": d,
                "gross": gross_return,
                "net": net_return,
                "turnover": turnover,
            })
        
        if len(daily_records) == 0:
            continue
            
        perf_df = pd.DataFrame(daily_records)
        r_gross = perf_df["gross"].to_numpy()
        r_net = perf_df["net"].to_numpy()
        
        ann_gross = float(np.mean(r_gross) * 252.0)
        ann_net = float(np.mean(r_net) * 252.0)
        ann_vol = float(np.std(r_gross) * np.sqrt(252.0)) if len(r_gross) > 1 else 1e-6
        sharpe_gross = float(ann_gross / ann_vol) if ann_vol > 0 else 0.0
        sharpe_net = float(ann_net / ann_vol) if ann_vol > 0 else 0.0
        
        cum_net = np.cumprod(1.0 + r_net)
        peak = np.maximum.accumulate(cum_net)
        max_dd = float(np.min((cum_net - peak) / peak)) if len(cum_net) > 0 else 0.0
        avg_turnover = float(np.mean(perf_df["turnover"]))
        
        results[f"h{h}"] = {
            "horizon_days": h,
            "n_days": len(perf_df),
            "ann_gross_return_pct": round(ann_gross * 100, 2),
            "ann_net_return_pct": round(ann_net * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe_gross": round(sharpe_gross, 2),
            "sharpe_net": round(sharpe_net, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_turnover_pct": round(avg_turnover * 100, 2),
            "cost_drag_pct": round(2 * avg_turnover * cost_bps / 100, 2),
        }
    
    return results


def run_robustness_suite():
    """Execute full robustness test suite."""
    master_path = DATA_FINAL / "master_dataset.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Master panel not found at {master_path}")
    
    df = pd.read_parquet(master_path)
    df = df.dropna(subset=["net_signal", "ret_fwd_1d"]).reset_index(drop=True)
    
    # Filter for minimum articles
    df = df[df["article_count_total"] >= 3].copy()
    
    signals_to_test = [
        ("net_signal", "net"),
        ("organic_signal", "organic"),
        ("sponsored_signal", "sponsored"),
        ("signal_organic_closed", "organic_closed"),
        ("signal_sponsored_closed", "sponsored_closed"),
    ]
    
    all_results = {}
    all_p_values = []
    all_test_names = []
    
    for signal_col, label in signals_to_test:
        if signal_col not in df.columns:
            logger.warning(f"Signal {signal_col} not found, skipping")
            continue
        
        logger.info(f"Running robustness tests for {signal_col}...")
        
        # Placebo: shuffled dates
        placebo_dates = placebo_shuffled_dates(df, signal_col)
        all_results[f"{label}_placebo_dates"] = placebo_dates
        all_p_values.append(placebo_dates["p_value_twotailed"])
        all_test_names.append(f"{label}: placebo_dates")
        
        # Placebo: shuffled tickers
        placebo_tickers = placebo_shuffled_tickers(df, signal_col)
        all_results[f"{label}_placebo_tickers"] = placebo_tickers
        all_p_values.append(placebo_tickers["p_value_twotailed"])
        all_test_names.append(f"{label}: placebo_tickers")
        
        # Permutation test
        perm_test = permutation_test_ic(df, signal_col)
        all_results[f"{label}_permutation"] = perm_test
        all_p_values.append(perm_test["p_value_twotailed"])
        all_test_names.append(f"{label}: permutation")
    
    # Global FDR across ALL robustness tests
    global_fdr = global_benjamini_hochberg(all_p_values)
    for i, name in enumerate(all_test_names):
        all_results[name]["p_value_fdr_global"] = round(global_fdr[i], 6)
    
    # Holding period sweep for net signal
    if "net_signal" in df.columns:
        logger.info("Running holding period sweep...")
        sweep = holding_period_sweep(df, "net_signal", max_horizon=20, cost_bps=10.0)
        all_results["holding_period_sweep"] = sweep
    
    # Save
    out_json = OUTPUTS_TABLES / "robustness_tests.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Robustness tests saved to {out_json}")
    
    # Summary print
    print("\n=== ROBUSTNESS TEST SUMMARY ===")
    for name in all_test_names:
        r = all_results[name]
        print(f"{name}: IC={r.get('original_mean_ic', r.get('perm_mean', 'N/A')):.4f}, "
              f"p={r.get('p_value', r.get('p_value_twotailed', 'N/A')):.4f}, "
              f"FDR_global={r.get('p_value_fdr_global', 'N/A'):.4f}")


if __name__ == "__main__":
    run_robustness_suite()