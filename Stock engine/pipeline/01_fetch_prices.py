"""Pipeline Step 1 — Fetch NIFTY 50 OHLCV prices and compute returns.

Downloads daily OHLCV data from Yahoo Finance for all NIFTY 50 constituents
and the NIFTY 50 index itself.  Computes three return types per stock:

    ret_overnight   = open_t / close_{t-1} - 1
    ret_intraday    = close_t / open_t - 1
    ret_close2close = close_t / close_{t-1} - 1

Saves:
    data/raw/prices/{TICKER}.parquet   — per-ticker OHLCV files
    data/processed/returns.parquet     — merged long-format return panel

Usage:
    python pipeline/01_fetch_prices.py                # full run
    python pipeline/01_fetch_prices.py --dry-run      # 3 tickers, 30 days
    python pipeline/01_fetch_prices.py --tickers RELIANCE.NS TCS.NS
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROCESSED,
    FETCH_END,
    FETCH_START,
    NUMPY_SEED,
    PYTHON_SEED,
    RAW_PRICES,
    seed_everything,
    setup_logging,
    NIFTY50_INDEX_TICKER,
    CAP_TIER_TOP10,
    CAP_TIER_MID,
)
from config.nifty50_tickers import get_ticker_list, get_cap_tier_map, get_sector_map

logger = setup_logging()
seed_everything()

# ── constants ─────────────────────────────────────────────────────
DRY_RUN_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
DRY_RUN_DAYS = 30


# ── helpers ───────────────────────────────────────────────────────

def download_ticker(
    ticker: str, start: str, end: str, max_retries: int = 3
) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker from Yahoo Finance.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by Date, or None on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {ticker} (attempt {attempt}/{max_retries})")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # yfinance sometimes returns MultiIndex columns; flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only the columns we need
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.warning(f"{ticker}: missing columns {missing}")
                return None

            df = df[required_cols].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df = df.sort_index()

            logger.info(f"{ticker}: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
            return df

        except Exception as e:
            logger.error(f"{ticker} attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logger.error(f"Giving up on {ticker}")
                return None


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return columns to an OHLCV DataFrame.

    Returns
    -------
    DataFrame with additional columns:
        ret_overnight   : open_t / close_{t-1} - 1
        ret_intraday    : close_t / open_t - 1
        ret_close2close : close_t / close_{t-1} - 1
    """
    df = df.copy()
    prev_close = df["Close"].shift(1)

    df["ret_overnight"] = df["Open"] / prev_close - 1
    df["ret_intraday"] = df["Close"] / df["Open"] - 1
    df["ret_close2close"] = df["Close"] / prev_close - 1

    # First row has no previous close → NaN returns; keep it for OHLCV
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info(f"Saved {path}  ({len(df)} rows, {path.stat().st_size / 1024:.0f} KB)")


# ── main pipeline ─────────────────────────────────────────────────

def run(
    tickers: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Execute the price-fetch pipeline.

    Parameters
    ----------
    tickers : list[str] | None
        Specific tickers to download.  None → full NIFTY 50 + index.
    dry_run : bool
        If True, limit to DRY_RUN_TICKERS and DRY_RUN_DAYS.
    """
    # ── resolve ticker list ───────────────────────────────────────
    if tickers:
        ticker_list = tickers
    elif dry_run:
        ticker_list = DRY_RUN_TICKERS
    else:
        ticker_list = get_ticker_list()

    # Always include the NIFTY 50 index for market-regime computation
    if NIFTY50_INDEX_TICKER not in ticker_list:
        ticker_list = [NIFTY50_INDEX_TICKER] + ticker_list

    # ── resolve date range ────────────────────────────────────────
    if dry_run:
        # Align with GDELT dry run (Jan 2024) plus buffers
        start_str = "2023-12-01"
        end_str = "2024-02-15"
        logger.info(f"DRY RUN: {len(ticker_list)} tickers, {start_str} → {end_str}")
    else:
        start_str = FETCH_START
        end_str = FETCH_END
        logger.info(f"FULL RUN: {len(ticker_list)} tickers, {start_str} → {end_str}")

    # ── download and save per-ticker files ────────────────────────
    cap_tier_map = get_cap_tier_map()
    sector_map = get_sector_map()
    all_returns: list[pd.DataFrame] = []
    success_count = 0
    fail_count = 0

    for ticker in ticker_list:
        df = download_ticker(ticker, start_str, end_str)
        if df is None:
            fail_count += 1
            continue

        # Save raw OHLCV
        safe_name = ticker.replace("^", "IDX_")
        save_parquet(df, RAW_PRICES / f"{safe_name}.parquet")

        # Compute returns
        df_ret = compute_returns(df)
        df_ret["ticker"] = ticker
        df_ret["sector"] = sector_map.get(ticker, "INDEX")
        df_ret["cap_tier"] = cap_tier_map.get(ticker, "INDEX")
        all_returns.append(df_ret)
        success_count += 1

    logger.info(f"Downloaded {success_count} tickers, {fail_count} failures")

    if not all_returns:
        logger.error("No data downloaded — aborting")
        return

    # ── build merged return panel ─────────────────────────────────
    panel = pd.concat(all_returns, axis=0)
    panel = panel.reset_index()  # date becomes a column

    # Ensure date column is clean
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── add market regime (based on NIFTY 50 index) ───────────────
    idx_returns = panel.loc[
        panel["ticker"] == NIFTY50_INDEX_TICKER, ["date", "ret_close2close"]
    ].copy()
    idx_returns = idx_returns.sort_values("date").set_index("date")
    idx_returns["nifty_60d_return"] = (
        (1 + idx_returns["ret_close2close"]).rolling(60).apply(np.prod, raw=True) - 1
    )

    # Classify regime
    def _regime(ret60: float) -> str:
        if pd.isna(ret60):
            return "SIDEWAYS"
        if ret60 > 0.10:
            return "BULL"
        if ret60 < -0.10:
            return "BEAR"
        return "SIDEWAYS"

    idx_returns["market_regime"] = idx_returns["nifty_60d_return"].apply(_regime)
    regime_series = idx_returns["market_regime"]

    # Map regime back to panel
    panel = panel.merge(
        regime_series.reset_index().rename(columns={"date": "date"}),
        on="date",
        how="left",
    )
    panel["market_regime"] = panel["market_regime"].fillna("SIDEWAYS")

    # ── save ──────────────────────────────────────────────────────
    save_parquet(panel, DATA_PROCESSED / "returns.parquet")

    # Summary stats
    logger.info("=== PIPELINE 01 COMPLETE ===")
    logger.info(f"Tickers: {panel['ticker'].nunique()}")
    logger.info(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    logger.info(f"Total rows: {len(panel):,}")
    logger.info(f"Regime distribution:\n{panel['market_regime'].value_counts().to_string()}")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NIFTY 50 prices and compute returns."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Quick test: {len(DRY_RUN_TICKERS)} tickers, last {DRY_RUN_DAYS} days",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific ticker(s) to download, e.g. RELIANCE.NS TCS.NS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(tickers=args.tickers, dry_run=args.dry_run)
