"""Pipeline Step 4 — Sponsored News Scoring Engine.

Outputs:
  data/processed/sponsored_scores.parquet  — all articles scored with class & attribution
  outputs/figures/scoring_distribution.png — empirical score histogram with abstain band
  outputs/tables/scoring_summary.json     — class counts, coverage, and signal statistics

Usage:
  python pipeline/04_sponsored_scoring_engine.py
  python pipeline/04_sponsored_scoring_engine.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────
from config.settings import (
    DATA_PROCESSED,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    SPONSORED_PROB_HIGH,
    SPONSORED_PROB_LOW,
    seed_everything,
    setup_logging,
)
from pipeline.scoring_engine import (
    DEFAULT_BIAS,
    DEFAULT_WEIGHTS,
    score_dataframe,
    score_single_article,
)

logger = setup_logging()
seed_everything()


def plot_score_distribution(
    df: pd.DataFrame,
    save_path: Path,
    tau_low: float = SPONSORED_PROB_LOW,
    tau_high: float = SPONSORED_PROB_HIGH,
) -> None:
    """Generate and save the score distribution plot highlighting the Abstain Zone."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scores = df["sponsored_score"].dropna().values

    # Histogram of scores
    counts, bins, patches = ax.hist(
        scores,
        bins=50,
        range=(0.0, 1.0),
        color="#3b82f6",
        alpha=0.7,
        edgecolor="white",
    )

    # Highlight zones
    ax.axvspan(0.0, tau_low, color="#10b981", alpha=0.15, label=f"Organic Zone (score <= {tau_low:.2f})")
    ax.axvspan(tau_low, tau_high, color="#f59e0b", alpha=0.20, label=f"Abstain Zone ({tau_low:.2f} < score < {tau_high:.2f})")
    ax.axvspan(tau_high, 1.0, color="#ef4444", alpha=0.15, label=f"Sponsored Zone (score >= {tau_high:.2f})")

    ax.axvline(tau_low, color="#059669", linestyle="--", linewidth=1.5)
    ax.axvline(tau_high, color="#dc2626", linestyle="--", linewidth=1.5)

    ax.set_title("HHEE News Sponsoredness Score Distribution with Calibrated Abstain Zone", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sponsoredness Score S(A)", fontsize=11)
    ax.set_ylabel("Article Count", fontsize=11)
    ax.legend(loc="upper center", frameon=True, fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved score distribution plot: {save_path}")


def run(dry_run: bool = False) -> None:
    """Execute the mathematical scoring pipeline."""
    input_path = DATA_PROCESSED / "gdelt_labeled.parquet"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run pipeline/03_label_news.py first.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} articles from {input_path}")

    if dry_run:
        df = df.head(20_000).copy()
        logger.info(f"DRY RUN: Processing {len(df):,} rows")

    tau_low = SPONSORED_PROB_LOW
    tau_high = SPONSORED_PROB_HIGH

    scored_df = score_dataframe(df, tau_low=tau_low, tau_high=tau_high)

    # ── Save Parquet ──────────────────────────────────────────────────
    output_path = DATA_PROCESSED / "sponsored_scores.parquet"
    scored_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"Saved scored dataset to {output_path} ({len(scored_df):,} rows)")

    # ── Summary & Metrics ─────────────────────────────────────────────
    n_total = len(scored_df)
    n_sponsored = int((scored_df["sponsored_class"] == "sponsored").sum())
    n_organic = int((scored_df["sponsored_class"] == "organic").sum())
    n_unclear = int((scored_df["sponsored_class"] == "unclear").sum())
    coverage = (n_sponsored + n_organic) / max(n_total, 1)

    summary = {
        "total_articles": n_total,
        "n_sponsored": n_sponsored,
        "n_organic": n_organic,
        "n_unclear_abstain": n_unclear,
        "pct_sponsored": round((n_sponsored / n_total) * 100, 2),
        "pct_organic": round((n_organic / n_total) * 100, 2),
        "pct_unclear": round((n_unclear / n_total) * 100, 2),
        "effective_coverage_pct": round(coverage * 100, 2),
        "tau_low": tau_low,
        "tau_high": tau_high,
        "mean_score": round(float(scored_df["sponsored_score"].mean()), 4),
        "median_score": round(float(scored_df["sponsored_score"].median()), 4),
    }

    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_TABLES / "scoring_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved scoring summary to {summary_path}")

    # Plot
    plot_score_distribution(scored_df, OUTPUTS_FIGURES / "scoring_distribution.png", tau_low=tau_low, tau_high=tau_high)

    logger.info("=== HHEE SCORING PIPELINE COMPLETE ===")
    logger.info(f"  Sponsored: {n_sponsored:,} ({summary['pct_sponsored']}%)")
    logger.info(f"  Organic:   {n_organic:,} ({summary['pct_organic']}%)")
    logger.info(f"  Unclear:   {n_unclear:,} ({summary['pct_unclear']}%) [Abstain Zone]")
    logger.info(f"  Coverage:  {summary['effective_coverage_pct']}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the HHEE news scoring pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Process first 20,000 rows only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
