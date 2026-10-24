#!/usr/bin/env python
"""Quick demo runner using pre-cached demo data (~30 seconds)."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path("stock-engine/data/demo")

def check_demo_data():
    """Verify demo data exists."""
    required = [
        "prices_demo.parquet",
        "returns_demo.parquet",
        "gdelt_labeled_demo.parquet",
        "sponsored_scores_demo.parquet",
        "daily_signals_demo.parquet",
        "master_dataset_demo.parquet",
    ]
    missing = [f for f in required if not (DEMO_DIR / f).exists()]
    if missing:
        print(f"Missing demo files: {missing}")
        return False
    return True

def run_demo():
    """Run quick end-to-end demo with demo data."""
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr

    print("=" * 60)
    print("NIFTY 50 News-Signal Demo (Cached Data)")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading demo data...")
    prices = pd.read_parquet(DEMO_DIR / "prices_demo.parquet")
    returns = pd.read_parquet(DEMO_DIR / "returns_demo.parquet")
    gdelt = pd.read_parquet(DEMO_DIR / "gdelt_labeled_demo.parquet")
    scored = pd.read_parquet(DEMO_DIR / "sponsored_scores_demo.parquet")
    daily = pd.read_parquet(DEMO_DIR / "daily_signals_demo.parquet")
    master = pd.read_parquet(DEMO_DIR / "master_dataset_demo.parquet")

    print(f"   Prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers")
    print(f"   GDELT articles: {len(gdelt):,}")
    print(f"   Daily signals: {len(daily):,}")
    print(f"   Master panel: {len(master):,} rows")

    # 2. Show signal distribution
    print("\n2. Signal distribution (demo data):")
    for col in ["net_signal", "organic_signal", "sponsored_signal"]:
        if col in daily.columns:
            print(f"   {col}: mean={daily[col].mean():.4f}, std={daily[col].std():.4f}")

    # 3. Quick IC test
    print("\n3. Quick IC test (signal -> next-day return):")
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(["ticker", "date"])

    # Use net_signal -> fwd1_ret_close2close
    if "net_signal" in master.columns and "ret_fwd_1d" in master.columns:
        valid = master[["net_signal", "ret_fwd_1d"]].dropna()
        if len(valid) > 10:
            ic, p = spearmanr(valid["net_signal"], valid["ret_fwd_1d"])
            print(f"   net_signal -> fwd1_close2close: IC={ic:.4f}, p={p:.4f}")

    # 4. Show propagation demo
    print("\n4. Propagation demo (from cached analysis):")
    try:
        import json
        with open("stock-engine/outputs/tables/propagation_cascade_analysis.json") as f:
            analysis = json.load(f)
        if analysis.get('available'):
            print(f"   Total cascades: {analysis['total_cascades']}")
            for k in ["pr_wire_origin", "editorial_origin"]:
                s = analysis[k]
                print(f"   {s['label']}: n={s['n']}, latency={s['mean_latency_mins']:.0f}min, attenuation={s['mean_attenuation']:.3f}")
    except:
        print("   Run `python stock-engine/research/propagation_measured.py` for full analysis")

    # 5. Show signal decay
    print("\n5. Signal decay (from cached analysis):")
    try:
        with open("stock-engine/outputs/tables/signal_decay_analysis.json") as f:
            decay = json.load(f)
        print(f"   Peak IC: {decay['mean_ic'][decay['peak_ic_horizon']-1]:.4f} at horizon {decay['peak_ic_horizon']}")
        print(f"   Half-life: {decay['estimated_half_life_days']} days")
    except:
        print("   Run `python stock-engine/research/signal_decay.py --rebuild-forwards` for full analysis")

    print("\n" + "=" * 60)
    print("Demo complete! To run full pipeline: `python run_all.py --dry-run`")
    print("To launch dashboard: `streamlit run stock-engine/app/app.py`")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    DEMO_DIR = Path("stock-engine/data/demo")

    required = [
        "prices_demo.parquet",
        "returns_demo.parquet",
        "gdelt_labeled_demo.parquet",
        "sponsored_scores_demo.parquet",
        "daily_signals_demo.parquet",
        "master_dataset_demo.parquet",
    ]
    missing = [f for f in required if not (DEMO_DIR / f).exists()]
    if missing:
        print(f"Missing demo files: {missing}")
        sys.exit(1)
    run_demo()