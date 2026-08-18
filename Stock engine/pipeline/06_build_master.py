"""Pipeline Step 6 — Build the master analysis-ready dataset.

Merges daily news signals with stock returns, adds market regime
and sector metadata to produce the final dataset for hypothesis testing.

Inputs:
    data/processed/daily_signals.parquet    (from step 05)
    data/processed/returns.parquet          (from step 01)

Outputs:
    data/final/master_dataset.parquet

Final schema:
    ticker, date, sector, cap_tier, market_regime,
    signal_organic_closed, signal_organic_open,
    signal_sponsored_closed, signal_sponsored_open,
    organic_signal, sponsored_signal, net_signal,
    article_count_total, article_count_organic, article_count_sponsored,
    ret_overnight, ret_intraday, ret_close2close

Usage:
    python pipeline/06_build_master.py
    python pipeline/06_build_master.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_FINAL,
    DATA_PROCESSED,
    MIN_ARTICLES_FOR_SIGNAL,
    seed_everything,
    setup_logging,
)
from config.nifty50_tickers import get_sector_map, get_cap_tier_map

logger = setup_logging()
seed_everything()


# ── main pipeline ─────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Merge signals with returns to build the master dataset."""

    # ── Load inputs ───────────────────────────────────────────────
    signals_path = DATA_PROCESSED / "daily_signals.parquet"
    returns_path = DATA_PROCESSED / "returns.parquet"

    for path in [signals_path, returns_path]:
        if not path.exists():
            logger.error(f"Input not found: {path}")
            sys.exit(1)

    signals = pd.read_parquet(signals_path)
    returns = pd.read_parquet(returns_path)

    logger.info(f"Loaded signals: {len(signals):,} rows, {signals['ticker'].nunique()} tickers")
    logger.info(f"Loaded returns: {len(returns):,} rows, {returns['ticker'].nunique()} tickers")

    # ── Normalize date columns ────────────────────────────────────
    signals["effective_date"] = pd.to_datetime(signals["effective_date"]).dt.normalize()
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()

    # ── Select return columns for merge ───────────────────────────
    return_cols = [
        "ticker", "date",
        "ret_overnight", "ret_intraday", "ret_close2close",
        "market_regime",
    ]
    # Add sector and cap_tier if present
    for col in ["sector", "cap_tier"]:
        if col in returns.columns:
            return_cols.append(col)

    returns_slim = returns[return_cols].copy()
    returns_slim = returns_slim.rename(columns={"date": "effective_date"})

    # ── Merge ─────────────────────────────────────────────────────
    logger.info("Merging signals with returns...")
    master = signals.merge(
        returns_slim,
        on=["ticker", "effective_date"],
        how="inner",  # only keep rows where both signals and returns exist
    )
    logger.info(f"Merged dataset: {len(master):,} rows")

    if len(master) == 0:
        logger.error("No rows after merge! Check ticker names and date alignment.")
        sys.exit(1)

    # ── Ensure sector/cap_tier from config if missing ─────────────
    sector_map = get_sector_map()
    cap_tier_map = get_cap_tier_map()

    if "sector" not in master.columns or master["sector"].isna().any():
        master["sector"] = master["ticker"].map(sector_map).fillna("UNKNOWN")
    if "cap_tier" not in master.columns or master["cap_tier"].isna().any():
        master["cap_tier"] = master["ticker"].map(cap_tier_map).fillna("UNKNOWN")

    # Fill missing market_regime
    master["market_regime"] = master["market_regime"].fillna("SIDEWAYS")

    # ── Rename effective_date → date for clarity ──────────────────
    master = master.rename(columns={"effective_date": "date"})

    # ── Sort and reorder columns ──────────────────────────────────
    final_cols = [
        "ticker", "date", "sector", "cap_tier", "market_regime",
        "signal_organic_closed", "signal_organic_open",
        "signal_sponsored_closed", "signal_sponsored_open",
        "organic_signal", "sponsored_signal", "net_signal",
        "article_count_total", "article_count_organic", "article_count_sponsored",
        "ret_overnight", "ret_intraday", "ret_close2close",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in master.columns]
    master = master[final_cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Summary statistics ────────────────────────────────────────
    logger.info("=== MASTER DATASET SUMMARY ===")
    logger.info(f"  Shape: {master.shape}")
    logger.info(f"  Tickers: {master['ticker'].nunique()}")
    logger.info(f"  Date range: {master['date'].min().date()} → {master['date'].max().date()}")
    logger.info(f"  Rows with >= {MIN_ARTICLES_FOR_SIGNAL} articles: "
                f"{(master['article_count_total'] >= MIN_ARTICLES_FOR_SIGNAL).sum():,}")

    logger.info(f"\nMarket regime distribution:\n{master['market_regime'].value_counts().to_string()}")
    logger.info(f"\nSector distribution:\n{master['sector'].value_counts().to_string()}")
    logger.info(f"\nCap tier distribution:\n{master['cap_tier'].value_counts().to_string()}")

    # Signal coverage
    for col in ["signal_organic_closed", "signal_organic_open",
                "signal_sponsored_closed", "signal_sponsored_open"]:
        if col in master.columns:
            n_valid = master[col].notna().sum()
            logger.info(f"  {col}: {n_valid:,} non-null ({100*n_valid/len(master):.1f}%)")

    # Return stats
    for ret_col in ["ret_overnight", "ret_intraday", "ret_close2close"]:
        if ret_col in master.columns:
            logger.info(f"  {ret_col}: mean={master[ret_col].mean():.5f}, "
                        f"std={master[ret_col].std():.5f}")

    # ── Save ──────────────────────────────────────────────────────
    output_path = DATA_FINAL / "master_dataset.parquet"
    master.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"\nSaved {output_path}")
    logger.info(f"  {len(master):,} rows, {output_path.stat().st_size / 1024:.0f} KB")
    logger.info("=== PIPELINE 06 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build master analysis-ready dataset by merging signals with returns."
    )
    parser.add_argument("--dry-run", action="store_true", help="No effect (all data is merged)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
