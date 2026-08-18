r"""Pipeline Step 2 — Fetch GDELT 2.0 GKG data for NIFTY 50 companies via BigQuery.

Queries the GDELT Global Knowledge Graph (GKG) table in Google BigQuery
for articles mentioning NIFTY 50 company names or ticker keywords from
known Indian financial news domains.

Saves:
    data/raw/gdelt/gdelt_{YYYY_MM}.parquet   — chunked by month

Usage:
    python pipeline/02_fetch_gdelt.py --start-month 2020-01 --end-month 2026-03
    python pipeline/02_fetch_gdelt.py --month 2024-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROCESSED,
    FETCH_END,
    FETCH_START,
    GCP_PROJECT_ID,
    GDELT_TABLE,
    MAX_BYTES_BILLED,
    RAW_GDELT,
    TIMEZONE,
    seed_everything,
    setup_logging,
)
from config.nifty50_tickers import NIFTY50_TICKERS, get_keywords_map

logger = setup_logging()
seed_everything()

# ── constants ─────────────────────────────────────────────────────
DRY_RUN_COMPANIES = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
DRY_RUN_MONTH = "2024-01"

INDIA_DOMAIN_PATTERN = (
    r"economictimes|livemint|business-standard|moneycontrol|"
    r"ndtv|thehindu|financialexpress|zeebiz|cnbctv18|"
    r"businesstoday|reuters\.com/.*india|bloomberg\.com/.*india|"
    r"prnewswire|businesswire|globenewswire|ani-prsolutions|"
    r"newswire\.in|indiaprwire|businesswireindia|newsvoir|"
    r"firstpost|theprint|scroll\.in|thewire\.in|"
    r"outlookbusiness|forbesindia|freepressjournal|"
    r"tickertape|screener|trendlyne|equitymaster|"
    r"\.in/"
)


# ── helpers ───────────────────────────────────────────────────────

def _build_company_pattern(tickers: list[str]) -> str:
    """Build a BigQuery REGEXP pattern from company keywords."""
    keywords_map = get_keywords_map()
    all_keywords: list[str] = []
    for ticker in tickers:
        kws = keywords_map.get(ticker, [])
        all_keywords.extend(kw.lower().replace("&", "\\\\&") for kw in kws)
    seen = set()
    unique = []
    for kw in all_keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return "|".join(unique)


def _build_query(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    company_pattern: str,
) -> str:
    """Build the BigQuery SQL query string."""
    start_dt = f"{start_yyyymmdd[:4]}-{start_yyyymmdd[4:6]}-{start_yyyymmdd[6:8]}"
    end_dt = f"{end_yyyymmdd[:4]}-{end_yyyymmdd[4:6]}-{end_yyyymmdd[6:8]}"
    return f"""
SELECT
    DATE,
    SourceCommonName,
    DocumentIdentifier,
    V2Tone,
    V2Organizations
FROM
    `{GDELT_TABLE}`
WHERE
    _PARTITIONTIME >= TIMESTAMP('{start_dt}')
    AND _PARTITIONTIME <= TIMESTAMP('{end_dt}')
    AND DATE >= {start_yyyymmdd}000000
    AND DATE <= {end_yyyymmdd}235959
    AND (
        REGEXP_CONTAINS(
            LOWER(DocumentIdentifier),
            r"{INDIA_DOMAIN_PATTERN}"
        )
    )
    AND (
        REGEXP_CONTAINS(
            LOWER(IFNULL(V2Organizations, '')),
            r"{company_pattern}"
        )
        OR REGEXP_CONTAINS(
            LOWER(DocumentIdentifier),
            r"{company_pattern}"
        )
    )
"""


def _parse_v2tone(v2tone_str: str | None) -> dict[str, float]:
    """Parse GDELT V2Tone field into component scores."""
    defaults = {
        "tone": 0.0,
        "positive_score": 0.0,
        "negative_score": 0.0,
        "polarity": 0.0,
        "activity_ref_density": 0.0,
        "self_group_ref_density": 0.0,
        "word_count": 0.0,
    }
    if not v2tone_str or not isinstance(v2tone_str, str):
        return defaults

    parts = v2tone_str.split(",")
    keys = list(defaults.keys())
    result = {}
    for i, key in enumerate(keys):
        try:
            result[key] = float(parts[i]) if i < len(parts) else 0.0
        except (ValueError, IndexError):
            result[key] = 0.0
    return result


def _parse_gdelt_date(date_val) -> pd.Timestamp | None:
    """Parse GDELT DATE field (int64 YYYYMMDDHHMMSS) to tz-aware IST timestamp."""
    try:
        date_str = str(int(date_val))
        dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt, tz="UTC").tz_convert(TIMEZONE)
    except (ValueError, TypeError):
        return None


def _generate_month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Generate (start_yyyymmdd, end_yyyymmdd) pairs for each month in range."""
    # Normalize start
    if len(start) == 7:  # YYYY-MM
        start_dt = pd.Timestamp(f"{start}-01")
    else:
        start_dt = pd.Timestamp(start).replace(day=1)
        
    # Normalize end
    if len(end) == 7:  # YYYY-MM
        end_dt = pd.Timestamp(f"{end}-01") + pd.offsets.MonthEnd(0)
    else:
        end_dt = pd.Timestamp(end) + pd.offsets.MonthEnd(0)

    months = pd.date_range(start_dt, end_dt, freq="MS")
    ranges = []
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        ranges.append(
            (month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d"))
        )
    return ranges


def _process_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse V2Tone and DATE columns in a raw GDELT DataFrame."""
    if df.empty:
        return df

    tone_records = df["V2Tone"].apply(_parse_v2tone)
    tone_df = pd.DataFrame(tone_records.tolist(), index=df.index)
    df = pd.concat([df, tone_df], axis=1)

    df["datetime_ist"] = df["DATE"].apply(_parse_gdelt_date)
    df = df.dropna(subset=["datetime_ist"])

    df["source_url"] = df["DocumentIdentifier"].astype(str)
    df["source_domain"] = (
        df["source_url"]
        .str.extract(r"https?://(?:www\.)?([^/]+)", expand=False)
        .str.lower()
    )

    return df


def _create_empty_processed_df() -> pd.DataFrame:
    """Create an empty DataFrame with the correct processed GDELT schema."""
    df = pd.DataFrame({
        "DATE": pd.Series(dtype="int64"),
        "SourceCommonName": pd.Series(dtype="str"),
        "DocumentIdentifier": pd.Series(dtype="str"),
        "V2Tone": pd.Series(dtype="str"),
        "V2Organizations": pd.Series(dtype="str"),
        "tone": pd.Series(dtype="float64"),
        "positive_score": pd.Series(dtype="float64"),
        "negative_score": pd.Series(dtype="float64"),
        "polarity": pd.Series(dtype="float64"),
        "activity_ref_density": pd.Series(dtype="float64"),
        "self_group_ref_density": pd.Series(dtype="float64"),
        "word_count": pd.Series(dtype="float64"),
        "datetime_ist": pd.Series(dtype="datetime64[ns, Asia/Kolkata]"),
        "source_url": pd.Series(dtype="str"),
        "source_domain": pd.Series(dtype="str"),
    })
    return df


# ── main pipeline ─────────────────────────────────────────────────

def run(
    start_month: str = "2020-01",
    end_month: str = "2026-03",
    month: str | None = None,
    tickers: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Execute GDELT query and fetch script."""
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error(
            "google-cloud-bigquery not installed. Run:\n"
            "  pip install google-cloud-bigquery pyarrow db-dtypes"
        )
        sys.exit(1)

    project = GCP_PROJECT_ID or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.error(
            "GCP_PROJECT_ID not set. Please check config/settings.py."
        )
        sys.exit(1)

    client = bigquery.Client(project=project)

    # Determine date range and parameters
    if dry_run:
        month_ranges = _generate_month_ranges(DRY_RUN_MONTH, DRY_RUN_MONTH)
        ticker_list = DRY_RUN_COMPANIES
        logger.info(f"DRY RUN: {DRY_RUN_MONTH}, {len(ticker_list)} companies")
    elif month:
        month_ranges = _generate_month_ranges(month, month)
        ticker_list = tickers or list(NIFTY50_TICKERS.keys())
        logger.info(f"SINGLE MONTH RUN: {month}, {len(ticker_list)} companies")
    else:
        month_ranges = _generate_month_ranges(start_month, end_month)
        ticker_list = tickers or list(NIFTY50_TICKERS.keys())
        logger.info(
            f"RANGE RUN: {start_month} -> {end_month} ({len(month_ranges)} months), "
            f"{len(ticker_list)} companies"
        )

    company_pattern = _build_company_pattern(ticker_list)
    logger.info(f"Company pattern length: {len(company_pattern)} chars")

    total_bytes = 0

    for start_ym, end_ym in month_ranges:
        year = start_ym[:4]
        m_str = start_ym[4:6]
        month_label = f"{year}_{m_str}"
        outpath = RAW_GDELT / f"gdelt_{year}_{m_str}.parquet"

        if outpath.exists():
            logger.info(f"Skipping {month_label} — already exists: {outpath}")
            try:
                existing = pd.read_parquet(outpath)
                row_count = len(existing)
                if "DocumentIdentifier" in existing.columns:
                    unique_count = existing["DocumentIdentifier"].nunique()
                elif "source_url" in existing.columns:
                    unique_count = existing["source_url"].nunique()
                else:
                    unique_count = row_count
                print(f"Saved {month_label}: {row_count} rows, {unique_count} unique articles")
            except Exception as e:
                logger.warning(f"Error reading existing file {outpath}: {e}")
            continue

        query = _build_query(start_ym, end_ym, company_pattern)
        logger.info(f"Querying {month_label}...")

        try:
            job_config = bigquery.QueryJobConfig(
                maximum_bytes_billed=MAX_BYTES_BILLED
            )
            query_job = client.query(query, job_config=job_config)
            df = query_job.to_dataframe()

            bytes_processed = query_job.total_bytes_processed or 0
            total_bytes += bytes_processed
            logger.info(
                f"  {month_label}: {len(df)} raw rows, "
                f"{bytes_processed / 1024**2:.1f} MB processed"
            )

            # Process the GDELT schema
            df = _process_raw_df(df)
            
            # Save DF (if empty, write empty schema)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            if df.empty:
                logger.warning(f"  {month_label}: no results")
                df = _create_empty_processed_df()
            
            df.to_parquet(outpath, engine="pyarrow", index=False)
            
            row_count = len(df)
            unique_count = df["DocumentIdentifier"].nunique() if not df.empty else 0
            print(f"Saved {month_label}: {row_count} rows, {unique_count} unique articles")

        except Exception as e:
            logger.error(f"  {month_label} FAILED: {e}")
            # Save empty schema to continue and support resume
            outpath.parent.mkdir(parents=True, exist_ok=True)
            empty_df = _create_empty_processed_df()
            empty_df.to_parquet(outpath, engine="pyarrow", index=False)
            print(f"Saved {month_label}: 0 rows, 0 unique articles")
        
        # 3 second sleep to respect BigQuery rate limits
        time.sleep(3)

    # Coverage summary calculation
    months_attempted = len(month_ranges)
    months_succeeded = 0
    months_empty = 0
    total_rows = 0

    for start_ym, end_ym in month_ranges:
        year = start_ym[:4]
        m_str = start_ym[4:6]
        outpath = RAW_GDELT / f"gdelt_{year}_{m_str}.parquet"
        if outpath.exists():
            try:
                m_df = pd.read_parquet(outpath)
                rc = len(m_df)
                total_rows += rc
                if rc > 0:
                    months_succeeded += 1
                else:
                    months_empty += 1
            except Exception:
                months_empty += 1
        else:
            months_empty += 1

    print("\n=== GDELT COVERAGE SUMMARY ===")
    print(f"Months Attempted: {months_attempted}")
    print(f"Months Succeeded: {months_succeeded}")
    print(f"Months Empty:     {months_empty}")
    print(f"Total Rows:       {total_rows}")
    logger.info("=== PIPELINE 02 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GDELT 2.0 GKG data for NIFTY 50 companies."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Quick test: {DRY_RUN_MONTH}, {len(DRY_RUN_COMPANIES)} companies",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default="2020-01",
        help="Start month YYYY-MM (default: 2020-01)",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        default="2026-03",
        help="End month YYYY-MM (default: 2026-03)",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Fetch a single month YYYY-MM (e.g. 2020-01)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific ticker(s) to query for",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        start_month=args.start_month,
        end_month=args.end_month,
        month=args.month,
        tickers=args.tickers,
        dry_run=args.dry_run,
    )
