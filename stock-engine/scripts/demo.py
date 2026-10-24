#!/usr/bin/env python3
"""Demo script for NIFTY 50 News-Signal Engine.

Runs in under 30 seconds on sample data. Prints a real IC result with sample size.
Note: This uses a tiny synthetic sample and is NOT evidence of predictive power.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def main():
    # Use sample data if available, otherwise create a minimal one
    sample_path = Path("data/final/sample_master_dataset.parquet")
    
    if not sample_path.exists():
        print("Sample data not found. Creating minimal demo data...")
        # Create minimal sample data inline
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                   'AXISBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS']
        
        rows = []
        for d in dates:
            for t in tickers:
                rows.append({
                    'ticker': t,
                    'date': d,
                    'net_signal': np.random.normal(0, 1),
                    'article_count_total': np.random.randint(3, 10),
                    'ret_fwd_1d': np.random.normal(0, 0.015),
                })
        df = pd.DataFrame(rows)
    else:
        df = pd.read_parquet(sample_path)
    
    # Filter for valid data
    df = df.dropna(subset=['net_signal', 'ret_fwd_1d'])
    df = df[df['article_count_total'] >= 3].copy()
    
    if len(df) == 0:
        print("ERROR: No valid data after filtering")
        return 1
    
    # Compute daily IC
    daily_ics = []
    for d in df['date'].unique():
        g = df[df['date'] == d]
        valid = g[['net_signal', 'ret_fwd_1d']].notna().all(axis=1)
        if valid.sum() >= 5:
            ic, _ = stats.spearmanr(g.loc[valid, 'net_signal'], g.loc[valid, 'ret_fwd_1d'])
            daily_ics.append(ic)
    
    daily_ics = np.array(daily_ics)
    mean_ic = float(np.nanmean(daily_ics))
    n_days = len(daily_ics)
    n_obs = len(df)
    
    print("=" * 60)
    print("NIFTY 50 News-Signal Demo")
    print("=" * 60)
    print(f"Sample IC (net_signal -> ret_fwd_1d): {mean_ic:.4f}")
    print(f"Sample size: {n_obs} observations over {n_days} days")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()
    print("NOTE: This uses a tiny synthetic/random sample.")
    print("      It is NOT evidence of predictive power.")
    print("      Run the full pipeline with real data for actual results.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())