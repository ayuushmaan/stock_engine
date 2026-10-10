r"""Pipeline Step 8 — Phase 0.2: Honest evaluation on hand labels.

Retrains LightGBM with ALL circular features excluded:
    EXCLUDED: source_tier, tone, positive_score, negative_score,
              p_n_ratio, is_pr_wire, has_promo_ratio, has_risk_words

    RETAINED: tone_30d_variance, cross_seed_count, mention_burst_score,
              word_count, polarity

Evaluates ONLY on held-out hand labels (never seen during training).
Reports honest AUC and applies Gate 0 decision logic.

Inputs:
    data/final/hand_labels_workbook.csv  — completed by human annotator
    data/processed/sponsored_scores.parquet  — full feature matrix

Outputs:
    outputs/tables/honest_eval_report.json
    outputs/figures/honest_eval/  — ROC, PR, calibration plots
    models/honest_classifier.pkl  — retrained model (non-circular)

Gate 0 Decision Table:
    AUC ≥ 0.75   → Real detectable signal in metadata alone. Proceed with H1.
    0.60 – 0.75  → Weak signal, needs text. H1 depends on Phase 2.
    < 0.60       → Metadata can't do this. Pivot.

Usage:
    python pipeline/08_honest_eval.py
    python pipeline/08_honest_eval.py --csv data/final/hand_labels_workbook.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
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
    roc_curve,
    confusion_matrix,
)

# ── project imports ───────────────────────────────────────────────
from config.settings import (
    CLASSIFIER_TRAIN_END,
    CLASSIFIER_TRAIN_START,
    DATA_FINAL,
    DATA_PROCESSED,
    LGBM_PARAMS,
    LIGHTGBM_SEED,
    MODELS_DIR,
    NUMPY_SEED,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    seed_everything,
    setup_logging,
)

logger = setup_logging()
seed_everything(NUMPY_SEED)

# ── Feature sets ──────────────────────────────────────────────────
# Features that participate in the labeling function (MUST BE EXCLUDED)
CIRCULAR_FEATURES = [
    "source_tier",
    "tone_score",         # tone is used in label rules (threshold + sign)
    "positive_score",     # linear decomposition of tone
    "negative_score",     # linear decomposition of tone
    "p_n_ratio",          # nonlinear transform of tone components
    "is_pr_wire",         # exact match with labeling Rule 1
    "has_promo_ratio",    # partial match with labeling Rule 2 (URL patterns)
    "has_risk_words",     # exact match with labeling Rule 5
]

# Features that are structurally independent of the labeling function
HONEST_FEATURES = [
    "tone_30d_variance",
    "cross_seed_count",
    "mention_burst_score",
    "word_count",
    "polarity",
]

FIG_DIR = OUTPUTS_FIGURES / "honest_eval"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ── Load and validate hand labels ────────────────────────────────

def load_hand_labels(csv_path: Path) -> pd.DataFrame:
    """Load and validate the human-annotated hand_labels_workbook.csv.

    Returns
    -------
    pd.DataFrame
        Rows with valid hand_label (0 or 1), excluding unclear (-1).
    """
    logger.info(f"Loading hand labels from {csv_path}...")

    if not csv_path.exists():
        logger.error(
            f"Hand labels file not found: {csv_path}\n"
            "  Run pipeline/07_sample_for_hand_labels.py first,\n"
            "  then manually annotate all 1,000 articles."
        )
        sys.exit(1)

    labels = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(labels):,} rows from workbook")

    # Validate required columns
    if "hand_label" not in labels.columns:
        logger.error("Missing 'hand_label' column in workbook")
        sys.exit(1)

    if "source_url" not in labels.columns:
        logger.error("Missing 'source_url' column in workbook")
        sys.exit(1)

    # Convert hand_label to numeric, drop empty
    labels["hand_label"] = pd.to_numeric(labels["hand_label"], errors="coerce")
    n_empty = labels["hand_label"].isna().sum()
    if n_empty > 0:
        logger.warning(f"  {n_empty} rows have empty/invalid hand_label (will be skipped)")

    n_sponsored = (labels["hand_label"] == 1).sum()
    n_organic = (labels["hand_label"] == 0).sum()
    n_unclear = (labels["hand_label"] == -1).sum()
    n_valid = n_sponsored + n_organic

    logger.info(f"  Sponsored (1): {n_sponsored}")
    logger.info(f"  Organic   (0): {n_organic}")
    logger.info(f"  Unclear  (-1): {n_unclear}")
    logger.info(f"  Valid for evaluation: {n_valid}")

    if n_valid < 50:
        logger.error(
            f"Only {n_valid} valid labels. Need at least 50 for a meaningful evaluation.\n"
            "  Please complete more annotations."
        )
        sys.exit(1)

    if n_sponsored < 10 or n_organic < 10:
        logger.warning(
            f"Class imbalance alert: {n_sponsored} sponsored, {n_organic} organic.\n"
            "  Results will have wide confidence intervals."
        )

    # Keep only definite labels (exclude -1 unclear)
    valid = labels[labels["hand_label"].isin([0, 1])].copy()
    return valid


# ── Retrain without circular features ────────────────────────────

def retrain_honest_model(
    train_df: pd.DataFrame,
    features: list[str],
) -> object:
    """Train LightGBM on weak labels using ONLY non-circular features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with weak_label and all feature columns.
    features : list[str]
        Feature column names (non-circular only).

    Returns
    -------
    lgb.Booster
        Trained model.
    """
    import lightgbm as lgb

    labelled = train_df[train_df["weak_label"].notna()].copy()

    # Time-based split for training
    train_mask = (
        (labelled["effective_date"] >= CLASSIFIER_TRAIN_START)
        & (labelled["effective_date"] <= CLASSIFIER_TRAIN_END)
    )

    train_data = labelled[train_mask]
    logger.info(f"Honest model training on {len(train_data):,} rows with {len(features)} features")
    logger.info(f"  Features: {features}")

    X_train = train_data[features].values
    y_train = train_data["weak_label"].values

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": LIGHTGBM_SEED,
    }

    # Class balance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0 and n_neg > 0:
        params["scale_pos_weight"] = n_neg / n_pos

    d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)

    model = lgb.train(
        params,
        d_train,
        num_boost_round=300,
        valid_sets=[d_train],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    return model


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_on_hand_labels(
    model,
    features: list[str],
    eval_df: pd.DataFrame,
    label_col: str = "hand_label",
) -> dict:
    """Evaluate model on hand-labeled data.

    Parameters
    ----------
    model : lgb.Booster
        Trained model.
    features : list[str]
        Feature columns.
    eval_df : pd.DataFrame
        Hand-labeled evaluation data with feature columns present.
    label_col : str
        Column name with ground-truth labels.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    X = eval_df[features].values
    y_true = eval_df[label_col].values.astype(int)
    y_prob = model.predict(X)

    # AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        logger.warning("Only one class present in hand labels — cannot compute AUC")
        auc = np.nan
        ap = np.nan

    # Binary predictions at 0.5 threshold
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "honest_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "honest_ap": round(float(ap), 4) if not np.isnan(ap) else None,
        "n_eval": int(len(y_true)),
        "n_sponsored": int((y_true == 1).sum()),
        "n_organic": int((y_true == 0).sum()),
        "confusion_matrix": cm.tolist(),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "y_prob_mean": round(float(y_prob.mean()), 4),
        "y_prob_std": round(float(y_prob.std()), 4),
    }

    if not np.isnan(auc):
        report = classification_report(
            y_true, y_pred,
            target_names=["organic", "sponsored"],
            output_dict=True,
        )
        metrics["classification_report"] = report

    return metrics, y_true, y_prob


# ── Plotting ──────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, auc_val, save_path):
    """ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Honest AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Honest ROC — Hand-Labeled Evaluation\n(Circular features excluded)")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pr(y_true, y_prob, ap_val, save_path):
    """Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color="#dc2626", lw=2, label=f"AP = {ap_val:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Honest Precision-Recall — Hand-Labeled Evaluation\n(Circular features excluded)")
    ax.legend(loc="best", fontsize=12)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


def plot_calibration(y_true, y_prob, save_path, n_bins=10):
    """Calibration plot (predicted vs observed probability)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means_pred = []
    bin_means_true = []
    bin_counts = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_pred.append(y_prob[mask].mean())
            bin_means_true.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", alpha=0.5, label="Perfect calibration")
    ax.scatter(bin_means_pred, bin_means_true, c="#10b981", s=60, zorder=5)
    ax.plot(bin_means_pred, bin_means_true, color="#10b981", lw=1.5, label="Model calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive (Hand-Labeled)")
    ax.set_title("Calibration Plot — Honest Model\n(Circular features excluded)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


# ── Gate 0 decision ───────────────────────────────────────────────

def gate_0_decision(auc: float | None) -> dict:
    """Apply Gate 0 decision logic.

    Returns
    -------
    dict
        Decision with recommendation.
    """
    if auc is None:
        return {
            "auc": None,
            "tier": "ERROR",
            "interpretation": "Could not compute AUC (single class or insufficient data)",
            "recommendation": "Complete more hand labels and retry",
        }
    elif auc >= 0.75:
        return {
            "auc": auc,
            "tier": "STRONG",
            "interpretation": "Real detectable signal in metadata alone",
            "recommendation": "Proceed with H1 as the headline. Strong result — genuinely publishable.",
        }
    elif auc >= 0.60:
        return {
            "auc": auc,
            "tier": "WEAK",
            "interpretation": "Weak signal, needs text features",
            "recommendation": (
                "Proceed, but H1 depends on Phase 2. "
                "Text-based features (NLP on article body) are now mandatory, not optional."
            ),
        }
    else:
        return {
            "auc": auc,
            "tier": "INSUFFICIENT",
            "interpretation": "Metadata cannot detect sponsored content",
            "recommendation": (
                "PIVOT required. Two options:\n"
                "  Pivot A (recommended): Narrow to PR-wire attribution only — "
                "unambiguous, verifiable label. Drop the classifier entirely.\n"
                "  Pivot B: Make H2 (temporal asymmetry) the headline and demote H1 to a control."
            ),
        }


# ── main pipeline ─────────────────────────────────────────────────

def run(csv_path: str | None = None, mock_eval: bool = False) -> None:
    """Execute the honest evaluation pipeline."""

    # ── Resolve CSV path ──────────────────────────────────────────
    if csv_path:
        workbook_path = Path(csv_path)
    else:
        workbook_path = DATA_FINAL / "hand_labels_workbook.csv"

    # ── Load or generate mock hand labels ─────────────────────────
    if mock_eval:
        logger.info("Running in MOCK_EVAL mode for pipeline verification...")
        sample_parquet = DATA_FINAL / "hand_labels_sample.parquet"
        if not sample_parquet.exists():
            logger.error(f"Sample parquet not found: {sample_parquet}")
            sys.exit(1)
        mock_df = pd.read_parquet(sample_parquet).copy()
        # Synthetic ground truth for testing only
        rng = np.random.RandomState(42)
        mock_df["hand_label"] = rng.choice([0, 1], size=len(mock_df), p=[0.55, 0.45])
        mock_df["reason"] = "Mock annotation for pipeline testing"
        hand_labels = mock_df
    else:
        hand_labels = load_hand_labels(workbook_path)

    # ── Load full feature matrix ──────────────────────────────────
    scores_path = DATA_PROCESSED / "sponsored_scores.parquet"
    if not scores_path.exists():
        logger.error(f"Feature matrix not found: {scores_path}")
        sys.exit(1)

    logger.info(f"Loading full feature matrix from {scores_path}...")
    full_df = pd.read_parquet(scores_path)
    logger.info(f"Loaded {len(full_df):,} articles")

    # ── Join hand labels with feature matrix ──────────────────────
    # Match on source_url
    hand_urls = set(hand_labels["source_url"].dropna().values)
    eval_df = full_df[full_df["source_url"].isin(hand_urls)].copy()

    # Deduplicate (may have multiple tickers per URL)
    eval_df = eval_df.drop_duplicates(subset=["source_url"], keep="first")

    # Merge hand_label
    url_to_label = hand_labels.set_index("source_url")["hand_label"].to_dict()
    eval_df["hand_label"] = eval_df["source_url"].map(url_to_label)
    eval_df = eval_df[eval_df["hand_label"].notna()].copy()

    if len(eval_df) < 50:
        logger.warning(f"Only {len(eval_df)} articles matched in feature matrix. Using hand_labels directly with feature alignment.")
        eval_df = hand_labels.copy()
        for f in HONEST_FEATURES:
            if f not in eval_df.columns:
                eval_df[f] = 0.0

    logger.info(f"Matched {len(eval_df):,} hand-labeled articles with features")

    # ── Verify all honest features exist ──────────────────────────
    missing_features = [f for f in HONEST_FEATURES if f not in eval_df.columns]
    if missing_features:
        logger.error(f"Missing features in data: {missing_features}")
        logger.error("Re-run pipeline/04_sponsored_classifier.py to regenerate features")
        sys.exit(1)

    # ── Retrain model (non-circular features only) ────────────────
    logger.info("\n=== RETRAINING WITH NON-CIRCULAR FEATURES ===")
    model = retrain_honest_model(full_df, HONEST_FEATURES)

    # Save honest model
    model_path = MODELS_DIR / "honest_classifier.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved honest model: {model_path}")

    # ── Evaluate on hand labels ───────────────────────────────────
    logger.info("\n=== EVALUATING ON HAND LABELS ===")
    metrics, y_true, y_prob = evaluate_on_hand_labels(
        model, HONEST_FEATURES, eval_df, "hand_label"
    )

    # ── Also evaluate the ORIGINAL model on hand labels ───────────
    logger.info("\n=== EVALUATING ORIGINAL (CIRCULAR) MODEL ON HAND LABELS ===")
    original_model_path = MODELS_DIR / "sponsored_classifier.pkl"
    if original_model_path.exists():
        try:
            original_model = joblib.load(original_model_path)
        except Exception:
            import pickle
            with open(original_model_path, "rb") as f:
                original_model = pickle.load(f)
        orig_features = original_model.feature_name()
        available_orig = [f for f in orig_features if f in eval_df.columns]
        if len(available_orig) == len(orig_features):
            orig_metrics, _, _ = evaluate_on_hand_labels(
                original_model, orig_features, eval_df, "hand_label"
            )
            metrics["original_model_auc"] = orig_metrics["honest_auc"]
            metrics["original_model_ap"] = orig_metrics["honest_ap"]
            logger.info(f"  Original model AUC on hand labels: {orig_metrics['honest_auc']}")
        else:
            logger.warning(f"  Cannot evaluate original model — missing features: "
                         f"{set(orig_features) - set(available_orig)}")

    # ── Gate 0 ────────────────────────────────────────────────────
    decision = gate_0_decision(metrics["honest_auc"])
    metrics["gate_0_decision"] = decision

    # ── Plots ─────────────────────────────────────────────────────
    if metrics["honest_auc"] is not None:
        plot_roc(y_true, y_prob, metrics["honest_auc"], FIG_DIR / "honest_roc.png")
        plot_pr(y_true, y_prob, metrics["honest_ap"], FIG_DIR / "honest_pr.png")
        plot_calibration(y_true, y_prob, FIG_DIR / "honest_calibration.png")
        logger.info(f"Saved plots to {FIG_DIR}/")

    # ── Feature importance of honest model ────────────────────────
    importances = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": HONEST_FEATURES,
        "gain": importances,
    }).sort_values("gain", ascending=False)
    total_gain = imp_df["gain"].sum()
    imp_df["pct"] = 100 * imp_df["gain"] / total_gain if total_gain > 0 else 0
    metrics["honest_feature_importance"] = imp_df.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["feature"], imp_df["pct"], color="#10b981", edgecolor="white")
    ax.set_xlabel("Importance (% of Total Gain)")
    ax.set_title("Honest Model — Feature Importance\n(Circular features excluded)")
    ax.invert_yaxis()
    fig.savefig(FIG_DIR / "honest_feature_importance.png")
    plt.close(fig)

    # ── Save report ───────────────────────────────────────────────
    report = {
        "phase": "0.2",
        "description": "Honest evaluation on hand-labeled ground truth",
        "mock_mode": mock_eval,
        "run_timestamp": datetime.now().isoformat(),
        "excluded_features": CIRCULAR_FEATURES,
        "used_features": HONEST_FEATURES,
        "metrics": metrics,
        "gate_0_decision": decision,
        "python_version": platform.python_version(),
        "dataset_hash": hashlib.sha256(
            scores_path.read_bytes()[:4096]  # hash first 4KB for speed
        ).hexdigest()[:16],
    }

    report_path = OUTPUTS_TABLES / "honest_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Saved report: {report_path}")

    # ── Save hand labels as final parquet (if not mock) ───────────
    if not mock_eval:
        hand_parquet = DATA_FINAL / "hand_labels.parquet"
        eval_df.to_parquet(hand_parquet, engine="pyarrow", index=False)
        logger.info(f"Saved hand labels parquet: {hand_parquet}")

    # ── Print results ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"  PHASE 0.2 — HONEST EVALUATION RESULTS {'(MOCK TEST)' if mock_eval else ''}")
    logger.info("=" * 70)
    logger.info(f"  Honest AUC (non-circular features): {metrics['honest_auc']}")
    logger.info(f"  Honest AP  (non-circular features): {metrics['honest_ap']}")
    logger.info(f"  N evaluated:  {metrics['n_eval']}")
    logger.info(f"  N sponsored:  {metrics['n_sponsored']}")
    logger.info(f"  N organic:    {metrics['n_organic']}")
    if metrics.get("original_model_auc"):
        logger.info(f"  Original model AUC (for reference): {metrics['original_model_auc']}")
    logger.info("")
    logger.info(f"  ╔═══════════════════════════════════════════════════╗")
    logger.info(f"  ║  GATE 0 DECISION: {decision['tier']:>12}                  ║")
    logger.info(f"  ╠═══════════════════════════════════════════════════╣")
    logger.info(f"  ║  {decision['interpretation']:<50}║")
    logger.info(f"  ╚═══════════════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"  Recommendation:")
    for line in decision["recommendation"].split("\n"):
        logger.info(f"    {line}")
    logger.info("=" * 70)
    logger.info("=== PIPELINE 08 COMPLETE ===")


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0.2: Honest evaluation on hand-labeled ground truth."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to completed hand_labels_workbook.csv",
    )
    parser.add_argument(
        "--mock-eval", action="store_true",
        help="Run verification test using synthetic sample annotations without modifying hand_labels.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(csv_path=args.csv, mock_eval=args.mock_eval)
