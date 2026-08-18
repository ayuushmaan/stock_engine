"""Pipeline Step 4 — Train sponsored content classifier and score all articles.

Trains a LightGBM binary classifier on weak-labelled GDELT articles to
distinguish sponsored/PR content from organic editorial news.

Features engineered from GDELT metadata (no full text needed):
    TONE_SCORE          — raw GDELT tone value
    P_N_RATIO           — positive / (negative + 0.001)
    SOURCE_TIER         — 1/2/3 from domain lookup
    TONE_30D_VARIANCE   — rolling 30-day tone variance per source domain
    CROSS_SEED_COUNT    — articles with overlapping entities in 6hr window
    MENTION_BURST_SCORE — z-score of company mention count in 3hr window
    IS_PR_WIRE          — binary from PR wire domain list
    HAS_PROMO_RATIO     — ratio of promotional keywords in URL/title
    HAS_RISK_WORDS      — binary flag for risk/negative keywords in URL

Training:
    Train: 2020-01-01 to 2022-12-31  (weak-labelled rows only)
    Valid: 2023-01-01 to 2023-12-31

Outputs:
    models/sponsored_classifier.pkl       — saved LightGBM model
    data/processed/sponsored_scores.parquet  — all articles with sponsored_prob
    outputs/figures/feature_importance.png
    outputs/figures/precision_recall.png

Usage:
    python pipeline/04_sponsored_classifier.py
    python pipeline/04_sponsored_classifier.py --dry-run
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

# ── project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CLASSIFIER_TRAIN_END,
    CLASSIFIER_TRAIN_START,
    CLASSIFIER_VALID_END,
    CLASSIFIER_VALID_START,
    DATA_PROCESSED,
    LGBM_PARAMS,
    LIGHTGBM_SEED,
    MODELS_DIR,
    OUTPUTS_FIGURES,
    PROMO_KEYWORDS,
    PR_WIRE_DOMAINS,
    RISK_KEYWORDS,
    seed_everything,
    setup_logging,
)
from config.source_tiers import is_pr_wire

logger = setup_logging()
seed_everything()

# ── feature names ─────────────────────────────────────────────────
FEATURE_COLS = [
    "tone_score",
    "positive_score",
    "negative_score",
    "p_n_ratio",
    "source_tier",
    "tone_30d_variance",
    "cross_seed_count",
    "mention_burst_score",
    "is_pr_wire",
    "has_promo_ratio",
    "has_risk_words",
    "word_count",
    "polarity",
]


# ── feature engineering ───────────────────────────────────────────

def _compute_promo_ratio(url: str) -> float:
    """Fraction of promotional keywords found in URL string."""
    url_lower = url.lower()
    if not url_lower:
        return 0.0
    hits = sum(1 for kw in PROMO_KEYWORDS if kw in url_lower)
    return hits / len(PROMO_KEYWORDS)


def _has_risk_flag(url: str) -> int:
    """Binary flag: any risk keyword present in URL."""
    url_lower = url.lower()
    return int(any(kw in url_lower for kw in RISK_KEYWORDS))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all classifier features from the labelled GDELT DataFrame."""
    logger.info("Engineering features...")
    out = df.copy()

    # ── Basic tone features ───────────────────────────────────────
    out["tone_score"] = out["tone"].fillna(0).astype(float)
    out["positive_score"] = out.get("positive_score", pd.Series(0, index=out.index)).fillna(0).astype(float)
    out["negative_score"] = out.get("negative_score", pd.Series(0, index=out.index)).fillna(0).astype(float)
    out["p_n_ratio"] = out["positive_score"] / (out["negative_score"].abs() + 0.001)

    # Ensure word_count and polarity exist
    if "word_count" not in out.columns:
        out["word_count"] = 0.0
    if "polarity" not in out.columns:
        out["polarity"] = 0.0
    out["word_count"] = out["word_count"].fillna(0).astype(float)
    out["polarity"] = out["polarity"].fillna(0).astype(float)

    # ── Source-level features ─────────────────────────────────────
    out["is_pr_wire"] = out["source_url"].fillna("").apply(
        lambda url: int(is_pr_wire(url))
    )

    # ── URL-based keyword features ────────────────────────────────
    out["has_promo_ratio"] = out["source_url"].fillna("").apply(_compute_promo_ratio)
    out["has_risk_words"] = out["source_url"].fillna("").apply(_has_risk_flag)

    # ── Rolling tone variance per source domain (30-day window) ───
    logger.info("  Computing 30-day tone variance per domain...")
    if "source_domain" in out.columns and "datetime_ist" in out.columns:
        out = out.sort_values(["source_domain", "datetime_ist"])
        out["tone_30d_variance"] = (
            out.groupby("source_domain")["tone_score"]
            .transform(lambda x: x.rolling(window=30, min_periods=5).var())
        )
    else:
        out["tone_30d_variance"] = 0.0
    out["tone_30d_variance"] = out["tone_30d_variance"].fillna(0)

    # ── Cross-seeding count ───────────────────────────────────────
    # How many different domains covered "the same story" in a 6-hour window
    # Approximation: count distinct domains per effective_date per company
    logger.info("  Computing cross-seed counts...")
    if "effective_date" in out.columns and "source_domain" in out.columns:
        cross = (
            out.groupby(["effective_date"])["source_domain"]
            .transform("nunique")
        )
        out["cross_seed_count"] = cross
    else:
        out["cross_seed_count"] = 1
    out["cross_seed_count"] = out["cross_seed_count"].fillna(1).astype(float)

    # ── Mention burst score ───────────────────────────────────────
    # Z-score of article count in a 3-hour window for the effective date
    logger.info("  Computing mention burst scores...")
    if "effective_date" in out.columns:
        daily_counts = out.groupby("effective_date").size()
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        if std_count > 0:
            burst_map = ((daily_counts - mean_count) / std_count).to_dict()
        else:
            burst_map = {d: 0.0 for d in daily_counts.index}
        out["mention_burst_score"] = out["effective_date"].map(burst_map).fillna(0)
    else:
        out["mention_burst_score"] = 0.0

    logger.info(f"  Feature engineering complete. Shape: {out.shape}")
    return out


# ── model training ────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame) -> tuple:
    """Train LightGBM on weak labels, validate, return (model, metrics).

    Returns
    -------
    (model, train_metrics, valid_metrics)
    """
    import lightgbm as lgb

    # ── Split by date ─────────────────────────────────────────────
    labelled = df[df["weak_label"].notna()].copy()
    logger.info(f"Labelled rows: {len(labelled):,} "
                f"(sponsored={int((labelled['weak_label']==1).sum()):,}, "
                f"organic={int((labelled['weak_label']==0).sum()):,})")

    train_mask = (
        (labelled["effective_date"] >= CLASSIFIER_TRAIN_START) &
        (labelled["effective_date"] <= CLASSIFIER_TRAIN_END)
    )
    valid_mask = (
        (labelled["effective_date"] >= CLASSIFIER_VALID_START) &
        (labelled["effective_date"] <= CLASSIFIER_VALID_END)
    )

    train_df = labelled[train_mask]
    valid_df = labelled[valid_mask]

    logger.info(f"Train set: {len(train_df):,} rows | Valid set: {len(valid_df):,} rows")

    if len(train_df) < 100:
        logger.warning("Calendar split yielded insufficient train data (< 100 rows). Falling back to random split for dry-run/testing.")
        shuffled = labelled.sample(frac=1.0, random_state=LIGHTGBM_SEED)
        split_idx = int(len(shuffled) * 0.7)
        train_df = shuffled.iloc[:split_idx]
        valid_df = shuffled.iloc[split_idx:]
        logger.info(f"Random split - Train set: {len(train_df):,} rows | Valid set: {len(valid_df):,} rows")

        if len(train_df) < 10:
            logger.error("Not enough training data even after random split!")
            sys.exit(1)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["weak_label"].values
    X_valid = valid_df[FEATURE_COLS].values if len(valid_df) > 0 else None
    y_valid = valid_df["weak_label"].values if len(valid_df) > 0 else None

    # ── Train ─────────────────────────────────────────────────────
    params = LGBM_PARAMS.copy()
    params["random_state"] = LIGHTGBM_SEED

    # Handle class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0 and n_neg > 0:
        params["scale_pos_weight"] = n_neg / n_pos
        logger.info(f"  scale_pos_weight: {params['scale_pos_weight']:.2f}")

    # Extract callback params
    early_stopping = params.pop("early_stopping_rounds", 50)
    n_estimators = params.pop("n_estimators", 500)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)

    callbacks = [lgb.log_evaluation(period=100)]
    valid_sets = [train_data]
    valid_names = ["train"]

    if X_valid is not None and len(X_valid) > 0:
        valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=FEATURE_COLS)
        valid_sets.append(valid_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    # ── Metrics ───────────────────────────────────────────────────
    train_pred = model.predict(X_train)
    train_auc = roc_auc_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else 0
    train_ap = average_precision_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else 0
    logger.info(f"  Train AUC: {train_auc:.4f} | Train AP: {train_ap:.4f}")

    valid_metrics = {}
    if X_valid is not None and len(X_valid) > 0 and len(np.unique(y_valid)) > 1:
        valid_pred = model.predict(X_valid)
        valid_auc = roc_auc_score(y_valid, valid_pred)
        valid_ap = average_precision_score(y_valid, valid_pred)
        logger.info(f"  Valid AUC: {valid_auc:.4f} | Valid AP: {valid_ap:.4f}")
        valid_metrics = {"auc": valid_auc, "ap": valid_ap}

        # Classification report at 0.5 threshold
        logger.info("\n" + classification_report(
            y_valid, (valid_pred > 0.5).astype(int),
            target_names=["organic", "sponsored"],
        ))

    return model, {"auc": train_auc, "ap": train_ap}, valid_metrics


# ── visualization ─────────────────────────────────────────────────

def plot_feature_importance(model, save_path: Path) -> None:
    """Save feature importance bar chart."""
    import lightgbm as lgb

    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=15)
    ax.set_title("Sponsored Classifier — Feature Importance (Gain)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature importance plot: {save_path}")


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """Save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#2563eb", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Sponsored Classifier — Precision-Recall Curve", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved precision-recall plot: {save_path}")


def plot_tone_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Save tone distribution violin plot: high-sponsored vs low-sponsored."""
    high = df[df["sponsored_prob"] > 0.7]["tone_score"]
    low = df[df["sponsored_prob"] < 0.3]["tone_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(
        [low.dropna().values, high.dropna().values],
        positions=[1, 2],
        showmeans=True,
        showextrema=True,
    )
    # Color the violins
    colors = ["#2563eb", "#dc2626"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Organic\n(prob < 0.3)", "Sponsored\n(prob > 0.7)"], fontsize=12)
    ax.set_ylabel("GDELT Tone Score", fontsize=12)
    ax.set_title("Tone Distribution: Organic vs Sponsored Articles", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved tone distribution plot: {save_path}")


# ── main pipeline ─────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Execute the sponsored classifier pipeline."""

    # ── Load labelled data ────────────────────────────────────────
    input_path = DATA_PROCESSED / "gdelt_labeled.parquet"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run pipeline/03_label_news.py first.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} articles from {input_path}")

    if dry_run:
        df = df.head(20_000)
        logger.info(f"DRY RUN: trimmed to {len(df):,} rows")

    # ── Engineer features ─────────────────────────────────────────
    df = engineer_features(df)

    # ── Train classifier ──────────────────────────────────────────
    model, train_metrics, valid_metrics = train_classifier(df)

    # ── Save model ────────────────────────────────────────────────
    model_path = MODELS_DIR / "sponsored_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model: {model_path}")

    # ── Score ALL articles ────────────────────────────────────────
    logger.info("Scoring all articles...")
    X_all = df[FEATURE_COLS].values
    df["sponsored_prob"] = model.predict(X_all)
    df["sponsored_prob"] = df["sponsored_prob"].clip(0, 1)

    # ── Save scores ───────────────────────────────────────────────
    output_path = DATA_PROCESSED / "sponsored_scores.parquet"
    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"Saved {output_path} ({len(df):,} rows)")

    # ── Generate plots ────────────────────────────────────────────
    plot_feature_importance(model, OUTPUTS_FIGURES / "feature_importance.png")

    labelled = df[df["weak_label"].notna()]
    if len(labelled) > 0 and len(labelled["weak_label"].unique()) > 1:
        valid_mask = (
            (labelled["effective_date"] >= CLASSIFIER_VALID_START) &
            (labelled["effective_date"] <= CLASSIFIER_VALID_END)
        )
        valid_df = labelled[valid_mask]
        if len(valid_df) > 0 and len(valid_df["weak_label"].unique()) > 1:
            plot_precision_recall(
                valid_df["weak_label"].values,
                valid_df["sponsored_prob"].values,
                OUTPUTS_FIGURES / "precision_recall.png",
            )

    plot_tone_distribution(df, OUTPUTS_FIGURES / "tone_distribution.png")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=== SPONSORED SCORE SUMMARY ===")
    logger.info(f"  Mean prob:   {df['sponsored_prob'].mean():.3f}")
    logger.info(f"  Median prob: {df['sponsored_prob'].median():.3f}")
    logger.info(f"  >0.7 (sponsored): {(df['sponsored_prob']>0.7).sum():,}")
    logger.info(f"  <0.3 (organic):   {(df['sponsored_prob']<0.3).sum():,}")
    logger.info(f"  0.3-0.7 (grey):   {((df['sponsored_prob']>=0.3)&(df['sponsored_prob']<=0.7)).sum():,}")
    logger.info("=== PIPELINE 04 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sponsored content classifier and score all articles."
    )
    parser.add_argument("--dry-run", action="store_true", help="Process first 20k rows only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
