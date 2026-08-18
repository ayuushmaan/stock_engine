"""Pipeline Step 3 — Label GDELT news articles with time buckets and weak sponsored labels.

For every GDELT article:
  1. Assigns TIME_BUCKET  — CLOSED_POST / CLOSED_PRE / OPEN
  2. Assigns EFFECTIVE_DATE — the trading day this news "counts toward"
  3. Assigns WEAK_LABEL    — 1 (definite sponsored), 0 (definite organic), NaN (uncertain)

Inputs:
    data/raw/gdelt/gdelt_YYYY_MM.parquet   (from pipeline step 02)

Outputs:
    data/processed/gdelt_labeled.parquet

Usage:
    python pipeline/03_label_news.py
    python pipeline/03_label_news.py --dry-run          # first 10k rows only
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
    CLOSED_POST_START,
    CLOSED_PRE_END,
    DATA_PROCESSED,
    MARKET_CLOSE_IST,
    MARKET_OPEN_IST,
    OPEN_END,
    OPEN_START,
    PROMO_KEYWORDS,
    PR_WIRE_DOMAINS,
    RAW_GDELT,
    RISK_KEYWORDS,
    SPONSORED_URL_PATTERNS,
    TIMEZONE,
    seed_everything,
    setup_logging,
)
from config.source_tiers import (
    classify_source,
    is_pr_wire,
    ORGANIC_LABEL_DOMAINS,
)
from config.nifty50_tickers import get_keywords_map

logger = setup_logging()
seed_everything()

# ── NSE trading calendar ─────────────────────────────────────────

def _get_nse_trading_days(start: str = "2019-01-01", end: str = "2027-01-01") -> pd.DatetimeIndex:
    """Generate approximate NSE trading days (weekdays minus known holidays).

    Uses pandas business day calendar as a baseline. For production accuracy,
    this should be replaced with an actual NSE holiday calendar.
    """
    # Start with all business days
    bdays = pd.bdate_range(start=start, end=end)

    # Major NSE holidays (approximate — Republic Day, Holi, Good Friday,
    # Independence Day, Gandhi Jayanti, Diwali, Christmas etc.)
    # This is a simplification; the exact calendar changes yearly.
    # For research purposes, weekday-only is a reasonable approximation.
    return bdays


def _next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Find the next trading day on or after the given date."""
    # Normalize to date only
    date_only = date.normalize()
    future = trading_days[trading_days >= date_only]
    if len(future) > 0:
        return future[0]
    return date_only  # fallback


# ── time bucket assignment ────────────────────────────────────────

def assign_time_bucket(dt_ist: pd.Timestamp) -> str:
    """Classify a timestamp into CLOSED_POST, CLOSED_PRE, or OPEN.

    CLOSED_POST : 15:30 to 23:59 IST (after market close)
    CLOSED_PRE  : 00:00 to 09:15 IST (before market open)
    OPEN        : 09:15 to 15:30 IST (market hours)
    """
    t = dt_ist.time()
    open_time = pd.Timestamp(MARKET_OPEN_IST).time()
    close_time = pd.Timestamp(MARKET_CLOSE_IST).time()

    if t < open_time:
        return "CLOSED_PRE"
    elif t < close_time:
        return "OPEN"
    else:
        return "CLOSED_POST"


def assign_effective_date(
    dt_ist: pd.Timestamp,
    time_bucket: str,
    trading_days: pd.DatetimeIndex,
) -> pd.Timestamp:
    """Determine the effective trading date for a news article.

    CLOSED_POST → next trading day
    CLOSED_PRE  → same day if trading day, else next trading day
    OPEN        → same day if trading day, else next trading day
    """
    # Normalize and remove timezone to compare with tz-naive trading calendar
    date_only = dt_ist.normalize().tz_localize(None)

    if time_bucket == "CLOSED_POST":
        # News after close → affects NEXT trading day
        next_day = date_only + pd.Timedelta(days=1)
        return _next_trading_day(next_day, trading_days)
    else:
        # CLOSED_PRE or OPEN → affects same trading day (or next if holiday)
        return _next_trading_day(date_only, trading_days)


# ── weak label assignment ─────────────────────────────────────────

def _has_sponsored_url_pattern(url: str) -> bool:
    """Check if URL contains known sponsored content patterns."""
    url_lower = url.lower()
    return any(pat in url_lower for pat in SPONSORED_URL_PATTERNS)


def _has_risk_keywords(text: str) -> bool:
    """Check if text contains risk/negative keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in RISK_KEYWORDS)


def assign_weak_labels(df: pd.DataFrame) -> pd.Series:
    """Assign weak labels for the sponsored content classifier.

    Returns a Series with values:
        1.0  = definite sponsored
        0.0  = definite organic
        NaN  = uncertain (not used for training)

    Labelling rules:
        DEFINITE SPONSORED (label=1):
          - URL contains sponsored content patterns (/brandstudio/ etc.)
          - Source domain is a known PR wire
          - Tone in top 5% most positive AND NOT from a Tier-1 outlet

        DEFINITE ORGANIC (label=0):
          - Source is a Tier-1 quality outlet AND tone is negative or neutral (<=0)
          - Article URL/title contains risk keywords from a quality outlet
    """
    labels = pd.Series(np.nan, index=df.index, dtype="float64")

    # ── Definite Sponsored ────────────────────────────────────────
    # Rule 1: URL patterns
    has_sponsored_url = df["source_url"].fillna("").apply(_has_sponsored_url_pattern)
    labels[has_sponsored_url] = 1.0

    # Rule 2: PR wire domains
    is_pr = df["source_url"].fillna("").apply(is_pr_wire)
    labels[is_pr] = 1.0

    # Rule 3: Extremely positive tone from non-tier-1 sources
    tone_threshold = df["tone"].quantile(0.95)
    is_very_positive = df["tone"] >= tone_threshold
    is_not_tier1 = df["source_tier"] > 1
    labels[is_very_positive & is_not_tier1] = 1.0

    # ── Definite Organic ──────────────────────────────────────────
    # Rule 1: Negative/neutral tone from quality outlets
    is_organic_source = df["source_url"].fillna("").apply(
        lambda url: any(d in url.lower() for d in ORGANIC_LABEL_DOMAINS)
    )
    is_negative_neutral = df["tone"] <= 0
    labels[is_organic_source & is_negative_neutral] = 0.0

    # Rule 2: Contains risk keywords from quality outlets
    has_risk = df["source_url"].fillna("").apply(
        lambda url: _has_risk_keywords(url)
    )
    labels[is_organic_source & has_risk] = 0.0

    return labels


# ── main pipeline ─────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Execute the labelling pipeline."""

    # ── Load all GDELT parquet files ──────────────────────────────
    gdelt_files = sorted(RAW_GDELT.glob("gdelt_*.parquet"))
    if not gdelt_files:
        logger.error(f"No GDELT parquet files found in {RAW_GDELT}")
        logger.error("Run pipeline/02_fetch_gdelt.py first.")
        sys.exit(1)

    logger.info(f"Loading {len(gdelt_files)} GDELT files...")
    dfs = [pd.read_parquet(f) for f in gdelt_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total raw articles: {len(df):,}")

    if dry_run:
        df = df.head(10_000)
        logger.info(f"DRY RUN: trimmed to {len(df):,} rows")

    # ── Match articles to stock tickers ───────────────────────────
    logger.info("Matching articles to stock tickers...")
    keywords_map = get_keywords_map()
    
    def match_tickers(row) -> list[str]:
        matched = []
        v2orgs = str(row.get("V2Organizations", "")).lower()
        doc_id = str(row.get("DocumentIdentifier", "")).lower()
        for ticker, keywords in keywords_map.items():
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in v2orgs or kw_lower in doc_id:
                    matched.append(ticker)
                    break
        return matched

    df["matched_tickers"] = df.apply(match_tickers, axis=1)
    
    # Filter rows with no match
    original_len = len(df)
    df = df[df["matched_tickers"].apply(len) > 0].copy()
    logger.info(f"Filtered out {original_len - len(df):,} articles with no ticker match (remaining: {len(df):,})")
    
    # Explode DataFrame on matched_tickers to create a single 'ticker' column
    df = df.explode("matched_tickers")
    df = df.rename(columns={"matched_tickers": "ticker"})
    logger.info(f"Exploded articles by matched tickers. Total rows: {len(df):,}")

    # ── Ensure required columns exist ─────────────────────────────
    required_cols = ["datetime_ist", "source_url", "tone"]
    # If tone wasn't already parsed, try V2Tone
    if "tone" not in df.columns and "V2Tone" in df.columns:
        logger.info("Parsing V2Tone column...")
        tone_parts = df["V2Tone"].str.split(",", expand=True)
        df["tone"] = pd.to_numeric(tone_parts[0], errors="coerce").fillna(0)
        if tone_parts.shape[1] > 1:
            df["positive_score"] = pd.to_numeric(tone_parts[1], errors="coerce").fillna(0)
        if tone_parts.shape[1] > 2:
            df["negative_score"] = pd.to_numeric(tone_parts[2], errors="coerce").fillna(0)

    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            sys.exit(1)

    # ── Ensure datetime_ist is tz-aware ───────────────────────────
    if df["datetime_ist"].dt.tz is None:
        df["datetime_ist"] = df["datetime_ist"].dt.tz_localize(TIMEZONE)

    # ── Assign source tier ────────────────────────────────────────
    logger.info("Assigning source tiers...")
    df["source_tier"] = df["source_url"].fillna("").apply(classify_source)

    # ── Assign time buckets ───────────────────────────────────────
    logger.info("Assigning time buckets...")
    df["time_bucket"] = df["datetime_ist"].apply(assign_time_bucket)

    # ── Assign effective dates ────────────────────────────────────
    logger.info("Computing effective dates...")
    trading_days = _get_nse_trading_days()
    df["effective_date"] = df.apply(
        lambda row: assign_effective_date(row["datetime_ist"], row["time_bucket"], trading_days),
        axis=1,
    )
    # Normalize effective_date to date-only
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.normalize()

    # ── Assign weak labels ────────────────────────────────────────
    logger.info("Assigning weak labels...")
    df["weak_label"] = assign_weak_labels(df)

    # ── Summary statistics ────────────────────────────────────────
    n_sponsored = (df["weak_label"] == 1).sum()
    n_organic = (df["weak_label"] == 0).sum()
    n_uncertain = df["weak_label"].isna().sum()

    logger.info("=== LABEL SUMMARY ===")
    logger.info(f"  Definite sponsored:  {n_sponsored:>8,}  ({100*n_sponsored/len(df):.1f}%)")
    logger.info(f"  Definite organic:    {n_organic:>8,}  ({100*n_organic/len(df):.1f}%)")
    logger.info(f"  Uncertain:           {n_uncertain:>8,}  ({100*n_uncertain/len(df):.1f}%)")
    logger.info(f"Time bucket distribution:\n{df['time_bucket'].value_counts().to_string()}")
    logger.info(f"Source tier distribution:\n{df['source_tier'].value_counts().to_string()}")

    # ── Save ──────────────────────────────────────────────────────
    outpath = DATA_PROCESSED / "gdelt_labeled.parquet"
    df.to_parquet(outpath, engine="pyarrow", index=False)
    logger.info(f"Saved {outpath}  ({len(df):,} rows, {outpath.stat().st_size / 1024:.0f} KB)")
    logger.info("=== PIPELINE 03 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label GDELT articles with time buckets and weak sponsored labels."
    )
    parser.add_argument("--dry-run", action="store_true", help="Process first 10k rows only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
