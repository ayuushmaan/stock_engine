"""Pipeline Step 5 — Aggregate daily signals per stock from scored articles.

For each (stock, effective_date, time_bucket) group, computes:
    - organic_signal    : tone weighted by (1 - sponsored_prob)
    - sponsored_signal  : tone weighted by sponsored_prob
    - net_signal        : organic_signal - sponsored_signal
    - article counts by category

Also constructs the 2×2 research cells:
    signal_organic_closed   — organic articles in closed window
    signal_organic_open     — organic articles in open window
    signal_sponsored_closed — sponsored articles in closed window
    signal_sponsored_open   — sponsored articles in open window

Inputs:
    data/processed/sponsored_scores.parquet

Outputs:
    data/processed/daily_signals.parquet

Usage:
    python pipeline/05_aggregate_signals.py
    python pipeline/05_aggregate_signals.py --dry-run
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
    DATA_PROCESSED,
    MIN_ARTICLES_FOR_SIGNAL,
    SPONSORED_PROB_HIGH,
    SPONSORED_PROB_LOW,
    TONE_NORMALIZER,
    seed_everything,
    setup_logging,
)

logger = setup_logging()
seed_everything()


# ── helpers ───────────────────────────────────────────────────────

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean, returning 0 if total weight is zero."""
    w = weights.fillna(0)
    v = values.fillna(0)
    total_w = w.sum()
    if total_w == 0:
        return 0.0
    return float((v * w).sum() / total_w)


def _is_closed_window(bucket: str) -> bool:
    """Return True if time bucket is a closed-window category."""
    return bucket in ("CLOSED_POST", "CLOSED_PRE")


# ── aggregate per stock per day ───────────────────────────────────

def aggregate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level scores to daily stock-level signals.

    Parameters
    ----------
    df : DataFrame
        Must contain: ticker (or matched_ticker), effective_date,
        time_bucket, tone_score, sponsored_prob

    Returns
    -------
    DataFrame with one row per (ticker, effective_date)
    """
    # ── Ensure required columns ───────────────────────────────────
    # Detect the ticker column
    ticker_col = None
    for candidate in ["ticker", "matched_ticker", "stock"]:
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        logger.warning("No ticker column found — creating a synthetic one from V2Organizations")
        df["ticker"] = "UNKNOWN"
        ticker_col = "ticker"

    # Normalize tone to approx [-1, 1]
    df["norm_tone"] = df["tone_score"] / TONE_NORMALIZER

    # Add window flag
    df["is_closed"] = df["time_bucket"].isin(["CLOSED_POST", "CLOSED_PRE"])

    # Compute organic and sponsored weights
    df["organic_weight"] = 1.0 - df["sponsored_prob"].fillna(0.5)
    df["sponsored_weight"] = df["sponsored_prob"].fillna(0.5)

    # Precompute weighted tone
    df["weighted_tone_org"] = df["norm_tone"] * df["organic_weight"]
    df["weighted_tone_spon"] = df["norm_tone"] * df["sponsored_weight"]

    # Masks for organic and sponsored classification cutoffs
    mask_org = df["sponsored_prob"] < SPONSORED_PROB_LOW
    mask_spon = df["sponsored_prob"] > SPONSORED_PROB_HIGH

    # 2x2 cells helper columns: closed vs open windows
    df["weight_org_closed"] = np.where(mask_org & df["is_closed"], df["organic_weight"], np.nan)
    df["wtone_org_closed"] = np.where(mask_org & df["is_closed"], df["weighted_tone_org"], np.nan)

    df["weight_org_open"] = np.where(mask_org & ~df["is_closed"], df["organic_weight"], np.nan)
    df["wtone_org_open"] = np.where(mask_org & ~df["is_closed"], df["weighted_tone_org"], np.nan)

    df["weight_spon_closed"] = np.where(mask_spon & df["is_closed"], df["sponsored_weight"], np.nan)
    df["wtone_spon_closed"] = np.where(mask_spon & df["is_closed"], df["weighted_tone_spon"], np.nan)

    df["weight_sopen"] = np.where(mask_spon & ~df["is_closed"], df["sponsored_weight"], np.nan)
    df["wtone_sopen"] = np.where(mask_spon & ~df["is_closed"], df["weighted_tone_spon"], np.nan)

    # Counts indicators
    df["is_org_count"] = np.where(mask_org, 1, 0)
    df["is_spon_count"] = np.where(mask_spon, 1, 0)

    # Vectorized Groupby
    g = df.groupby([ticker_col, "effective_date"])
    agg = g.agg(
        # Numerator and denominator sums for overall signals
        sum_wtone_org=("weighted_tone_org", "sum"),
        sum_weight_org=("organic_weight", "sum"),
        sum_wtone_spon=("weighted_tone_spon", "sum"),
        sum_weight_spon=("sponsored_weight", "sum"),

        # Article counts
        article_count_total=("tone_score", "size"),
        article_count_organic=("is_org_count", "sum"),
        article_count_sponsored=("is_spon_count", "sum"),

        # Numerator and denominator sums for cells
        sum_wtone_org_closed=("wtone_org_closed", "sum"),
        sum_weight_org_closed=("weight_org_closed", "sum"),

        sum_wtone_org_open=("wtone_org_open", "sum"),
        sum_weight_org_open=("weight_org_open", "sum"),

        sum_wtone_spon_closed=("wtone_spon_closed", "sum"),
        sum_weight_spon_closed=("weight_spon_closed", "sum"),

        sum_wtone_spon_open=("wtone_sopen", "sum"),
        sum_weight_spon_open=("weight_sopen", "sum"),
    ).reset_index()

    # Calculate final signals with zero-weight guards
    agg["organic_signal"] = np.where(
        agg["sum_weight_org"] > 0,
        agg["sum_wtone_org"] / agg["sum_weight_org"],
        0.0
    )
    agg["sponsored_signal"] = np.where(
        agg["sum_weight_spon"] > 0,
        agg["sum_wtone_spon"] / agg["sum_weight_spon"],
        0.0
    )
    agg["net_signal"] = agg["organic_signal"] - agg["sponsored_signal"]

    agg["signal_organic_closed"] = np.where(
        agg["sum_weight_org_closed"] > 0,
        agg["sum_wtone_org_closed"] / agg["sum_weight_org_closed"],
        np.nan
    )
    agg["signal_organic_open"] = np.where(
        agg["sum_weight_org_open"] > 0,
        agg["sum_wtone_org_open"] / agg["sum_weight_org_open"],
        np.nan
    )
    agg["signal_sponsored_closed"] = np.where(
        agg["sum_weight_spon_closed"] > 0,
        agg["sum_wtone_spon_closed"] / agg["sum_weight_spon_closed"],
        np.nan
    )
    agg["signal_sponsored_open"] = np.where(
        agg["sum_weight_spon_open"] > 0,
        agg["sum_wtone_spon_open"] / agg["sum_weight_spon_open"],
        np.nan
    )

    # Rename ticker column if necessary
    if ticker_col != "ticker":
        agg = agg.rename(columns={ticker_col: "ticker"})

    # Clean up and sort
    result = agg[[
        "ticker", "effective_date", "organic_signal", "sponsored_signal", "net_signal",
        "article_count_total", "article_count_organic", "article_count_sponsored",
        "signal_organic_closed", "signal_organic_open", "signal_sponsored_closed", "signal_sponsored_open"
    ]].copy()

    result["effective_date"] = pd.to_datetime(result["effective_date"])
    result = result.sort_values(["ticker", "effective_date"]).reset_index(drop=True)

    return result


# ── main pipeline ─────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Execute signal aggregation pipeline."""

    input_path = DATA_PROCESSED / "sponsored_scores.parquet"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run pipeline/04_sponsored_classifier.py first.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} scored articles")

    if dry_run:
        df = df.head(20_000)
        logger.info(f"DRY RUN: trimmed to {len(df):,} rows")

    # ── Ensure tone_score exists ──────────────────────────────────
    if "tone_score" not in df.columns:
        if "tone" in df.columns:
            df["tone_score"] = df["tone"]
        else:
            logger.error("No tone_score or tone column found!")
            sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────
    signals = aggregate_signals(df)
    logger.info(f"Aggregated to {len(signals):,} stock-day observations")

    # ── Coverage stats ────────────────────────────────────────────
    n_with_signal = (signals["article_count_total"] >= MIN_ARTICLES_FOR_SIGNAL).sum()
    logger.info(f"  Stock-days with >= {MIN_ARTICLES_FOR_SIGNAL} articles: {n_with_signal:,}")
    logger.info(f"  Unique tickers: {signals['ticker'].nunique()}")
    logger.info(f"  Date range: {signals['effective_date'].min()} → {signals['effective_date'].max()}")

    # 2×2 cell coverage
    for col in ["signal_organic_closed", "signal_organic_open",
                "signal_sponsored_closed", "signal_sponsored_open"]:
        n_valid = signals[col].notna().sum()
        logger.info(f"  {col}: {n_valid:,} non-null ({100*n_valid/len(signals):.1f}%)")

    # ── Save ──────────────────────────────────────────────────────
    output_path = DATA_PROCESSED / "daily_signals.parquet"
    signals.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"Saved {output_path} ({len(signals):,} rows, {output_path.stat().st_size / 1024:.0f} KB)")
    logger.info("=== PIPELINE 05 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate article scores to daily stock-level signals."
    )
    parser.add_argument("--dry-run", action="store_true", help="Process first 20k rows only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
