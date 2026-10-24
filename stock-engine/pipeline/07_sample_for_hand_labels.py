r"""Pipeline Step 7 — Phase 0.1: Stratified sample 1,000 articles for hand labeling.

Draws a stratified sample from gdelt_labeled.parquet, balanced across:
    - source_tier  (1, 2, 3)
    - tone_decile  (0–9, deciles of GDELT tone)
    - year         (2020–2026)

For each sampled article, includes metadata to assist the human reviewer:
    - source_url   (open this in browser to read the page)
    - source_domain, source_tier, tone, positive/negative scores
    - weak_label   (current heuristic label — DO NOT use this to decide)
    - sponsored_prob (current classifier score — DO NOT use this to decide)

The annotator fills in:
    - hand_label   : 1 = sponsored, 0 = organic, -1 = unclear
    - reason       : one-line reason for the label

Outputs:
    data/final/hand_labels_workbook.csv   — for human annotation (open in Excel/Sheets)
    data/final/hand_labels_sample.parquet — metadata for later joining

Usage:
    python pipeline/07_sample_for_hand_labels.py
    python pipeline/07_sample_for_hand_labels.py --n 500       # smaller sample
    python pipeline/07_sample_for_hand_labels.py --dry-run     # preview 20 articles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────
from config.settings import (
    DATA_FINAL,
    DATA_PROCESSED,
    NUMPY_SEED,
    seed_everything,
    setup_logging,
)

logger = setup_logging()
seed_everything(NUMPY_SEED)


# ── stratified sampling ──────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n_total: int = 1000,
    seed: int = NUMPY_SEED,
) -> pd.DataFrame:
    """Draw a stratified sample across source_tier × tone_decile × year.

    For performance, first draws a random oversample (50× target), then
    stratifies within that pool. This avoids computing on 1M+ rows.

    Parameters
    ----------
    df : pd.DataFrame
        Full labeled GDELT DataFrame (needs: source_tier, tone, datetime_ist or effective_date).
    n_total : int
        Target total number of articles to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled rows with stratum columns added.
    """
    logger.info(f"Starting stratified sampling (target: {n_total} from {len(df):,} rows)...")

    # ── Step 1: Pre-sample a manageable pool ──────────────────────
    pool_size = min(len(df), max(n_total * 50, 50_000))
    pool = df.sample(n=pool_size, random_state=seed).copy()
    logger.info(f"  Pre-sampled pool of {len(pool):,} rows for stratification")

    # ── Step 2: Build stratification columns ──────────────────────
    if "effective_date" in pool.columns:
        pool["year"] = pd.to_datetime(pool["effective_date"]).dt.year
    elif "datetime_ist" in pool.columns:
        pool["year"] = pool["datetime_ist"].dt.year
    else:
        raise ValueError("Need effective_date or datetime_ist for year stratification")

    # Tone decile (0-9) using numpy digitize for speed
    tone_vals = pool["tone"].fillna(0).values
    percentiles = np.percentile(tone_vals, np.arange(10, 100, 10))
    pool["tone_decile"] = np.digitize(tone_vals, percentiles)  # 0-9

    # ── Step 3: Sample from each stratum ─────────────────────────
    sampled_parts = []
    for _keys, group in pool.groupby(["source_tier", "tone_decile", "year"]):
        frac = n_total / pool_size
        target = max(1, int(round(len(group) * frac)))
        target = min(target, len(group))
        sampled_parts.append(group.sample(n=target, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Trim or pad to hit n_total
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n_total:
        deficit = n_total - len(sampled)
        used_indices = set(sampled.index)
        remaining = pool[~pool.index.isin(used_indices)]
        if len(remaining) > 0:
            extra = remaining.sample(n=min(deficit, len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    n_strata = pool.groupby(["source_tier", "tone_decile", "year"]).ngroups

    logger.info(
        f"  Sampled {len(sampled):,} articles across {n_strata} strata"
    )

    return sampled


# ── main pipeline ─────────────────────────────────────────────────

def run(n_sample: int = 1000, dry_run: bool = False) -> None:
    """Generate the hand-labeling workbook."""

    # ── Load data ─────────────────────────────────────────────────
    # Prefer sponsored_scores.parquet (has all features + sponsored_prob)
    scores_path = DATA_PROCESSED / "sponsored_scores.parquet"
    labeled_path = DATA_PROCESSED / "gdelt_labeled.parquet"

    if scores_path.exists():
        input_path = scores_path
    elif labeled_path.exists():
        input_path = labeled_path
    else:
        logger.error("No input data found. Run pipeline steps 03-04 first.")
        sys.exit(1)

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} articles")

    if dry_run:
        n_sample = 20
        logger.info(f"DRY RUN: sampling only {n_sample} articles")

    # ── Deduplicate by URL ────────────────────────────────────────
    # For hand-labeling, we want unique URLs (not exploded by ticker)
    if "source_url" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["source_url"], keep="first")
        logger.info(f"Deduplicated by URL: {before:,} → {len(df):,}")

    # ── Sample ────────────────────────────────────────────────────
    sample_df = stratified_sample(df, n_total=n_sample)

    # ── Build workbook ────────────────────────────────────────────
    # Columns the annotator needs to see
    context_cols = [
        "source_url",
        "source_domain",
        "source_tier",
        "tone",
        "positive_score",
        "negative_score",
        "polarity",
        "word_count",
        "time_bucket",
        "year",
        "tone_decile",
    ]

    # Include existing labels for reference (annotator should IGNORE these)
    reference_cols = []
    if "weak_label" in sample_df.columns:
        reference_cols.append("weak_label")
    if "sponsored_prob" in sample_df.columns:
        reference_cols.append("sponsored_prob")

    # Also keep ticker for context
    if "ticker" in sample_df.columns:
        context_cols.insert(1, "ticker")

    # Keep only columns that exist
    available_context = [c for c in context_cols if c in sample_df.columns]
    available_reference = [c for c in reference_cols if c in sample_df.columns]

    workbook = sample_df[available_context + available_reference].copy()

    # Add annotator columns (empty)
    workbook["hand_label"] = ""  # 1=sponsored, 0=organic, -1=unclear
    workbook["reason"] = ""     # one-line reason

    # Add a row ID for tracking
    workbook.insert(0, "sample_id", range(1, len(workbook) + 1))

    # ── Save workbook CSV ─────────────────────────────────────────
    csv_path = DATA_FINAL / "hand_labels_workbook.csv"
    workbook.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved annotation workbook: {csv_path}  ({len(workbook)} rows)")

    # ── Save sample parquet (full metadata) ───────────────────────
    parquet_path = DATA_FINAL / "hand_labels_sample.parquet"
    sample_df.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"Saved sample parquet: {parquet_path}")

    # ── Print summary ─────────────────────────────────────────────
    logger.info("\n=== SAMPLING SUMMARY ===")
    logger.info(f"  Total articles sampled: {len(workbook)}")
    logger.info(f"  Source tier distribution:")
    for tier, count in workbook["source_tier"].value_counts().sort_index().items():
        logger.info(f"    Tier {tier}: {count} ({100*count/len(workbook):.1f}%)")
    logger.info(f"  Year distribution:")
    for year, count in workbook["year"].value_counts().sort_index().items():
        logger.info(f"    {year}: {count} ({100*count/len(workbook):.1f}%)")
    logger.info(f"  Tone decile distribution:")
    for decile, count in workbook["tone_decile"].value_counts().sort_index().items():
        logger.info(f"    Decile {decile}: {count} ({100*count/len(workbook):.1f}%)")

    if "weak_label" in workbook.columns:
        wl_dist = workbook["weak_label"].value_counts(dropna=False)
        logger.info(f"  Current weak_label distribution in sample:")
        for label, count in wl_dist.items():
            label_str = {1.0: "sponsored", 0.0: "organic"}.get(label, "uncertain/NaN")
            logger.info(f"    {label_str}: {count} ({100*count/len(workbook):.1f}%)")

    logger.info(f"\n  NEXT STEPS:")
    logger.info(f"  1. Open {csv_path}")
    logger.info(f"  2. For each row, open the source_url in your browser")
    logger.info(f"  3. Read the page and fill in:")
    logger.info(f"       hand_label: 1=sponsored, 0=organic, -1=unclear")
    logger.info(f"       reason: one-line justification")
    logger.info(f"  4. Save the CSV")
    logger.info(f"  5. Run: python pipeline/08_honest_eval.py")
    logger.info(f"\n  Budget: ~1,000 articles × 25 sec/each ≈ 7 hours")
    logger.info(f"  Recommended: split over 3 evenings")
    logger.info("=== PIPELINE 07 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0.1: Stratified sample for hand labeling."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of articles to sample (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Sample only 20 articles for preview",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_sample=args.n, dry_run=args.dry_run)
