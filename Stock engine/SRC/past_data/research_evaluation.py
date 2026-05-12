import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from SRC.common.paths import HISTORY_DIR, RESEARCH_DIR, ensure_dirs

def _load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _file_sha256(path):
    if not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _join_history():
    ensure_dirs()
    sentiment = _load_csv(os.path.join(HISTORY_DIR, "nifty_sentiment_history.csv"))
    prices = _load_csv(os.path.join(HISTORY_DIR, "daily_prices_history.csv"))
    if sentiment.empty or prices.empty:
        return pd.DataFrame()

    if "Date" not in sentiment.columns or "Date" not in prices.columns:
        return pd.DataFrame()

    merged = pd.merge(sentiment, prices, on=["Date", "Symbol"], how="inner", suffixes=("_sent", "_price"))
    if merged.empty:
        return merged

    if "Sector_sent" in merged.columns:
        merged["Sector"] = merged["Sector_sent"]
    elif "Sector_price" in merged.columns:
        merged["Sector"] = merged["Sector_price"]
    else:
        merged["Sector"] = "UNKNOWN"

    merged["Pred_Score"] = pd.to_numeric(merged.get("Pred_Score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    merged["Actual_Pct_Change"] = pd.to_numeric(merged.get("Percent_Change"), errors="coerce").fillna(0.0)
    merged["Headline_Count"] = pd.to_numeric(merged.get("Headline_Count"), errors="coerce").fillna(0.0)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return merged


def _time_split(df, train_ratio=0.7):
    unique_dates = sorted(df["Date"].dropna().unique())
    if len(unique_dates) < 3:
        return pd.DataFrame(), pd.DataFrame(), None
    train_size = max(1, int(len(unique_dates) * train_ratio))
    train_dates = set(unique_dates[:train_size])
    test_dates = set(unique_dates[train_size:])
    cutoff = unique_dates[train_size - 1]
    train = df[df["Date"].isin(train_dates)].copy()
    test = df[df["Date"].isin(test_dates)].copy()
    return train, test, cutoff


def _calibrate_k(pred_score, actual_pct):
    denominator = float(np.dot(pred_score, pred_score))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(pred_score, actual_pct) / denominator)


def _direction_from_score(values, threshold=0.10):
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def _compute_metrics(y_true, y_pred_pct, pred_score, threshold=0.10):
    mae = float(np.mean(np.abs(y_pred_pct - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred_pct - y_true) ** 2)))
    mape_den = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_pred_pct - y_true) / mape_den)) * 100)

    if len(y_true) >= 3:
        pearson_corr, pearson_p = stats.pearsonr(pred_score, y_true)
        spearman_corr, spearman_p = stats.spearmanr(pred_score, y_true)
    else:
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan

    pred_dir = _direction_from_score(pred_score, threshold=threshold)
    true_dir = np.sign(y_true).astype(int)
    valid = pred_dir != 0
    if valid.sum() > 0:
        dir_accuracy = float((pred_dir[valid] == true_dir[valid]).mean())
        binom = stats.binomtest(int((pred_dir[valid] == true_dir[valid]).sum()), int(valid.sum()), p=0.5, alternative="greater")
        dir_p = float(binom.pvalue)
    else:
        dir_accuracy = np.nan
        dir_p = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "directional_accuracy": dir_accuracy,
        "directional_pvalue": dir_p,
    }


def _baseline_predictions(train, test, seed=42):
    rng = np.random.default_rng(seed)
    y_train = train["Actual_Pct_Change"].to_numpy(dtype=float)
    y_test = test["Actual_Pct_Change"].to_numpy(dtype=float)

    base = {}

    random_score = rng.uniform(-1.0, 1.0, size=len(test))
    random_k = _calibrate_k(random_score, y_test) if len(test) > 0 else 0.0
    base["random"] = (random_score, random_score * random_k)

    market_const = float(np.mean(y_train)) if len(y_train) else 0.0
    market_score = np.full(len(test), np.tanh(market_const / 2.0))
    market_pred = np.full(len(test), market_const)
    base["market_return"] = (market_score, market_pred)

    last_return = train.sort_values(["Symbol", "Date"]).groupby("Symbol")["Actual_Pct_Change"].last().to_dict()
    momentum_pred = test["Symbol"].map(last_return).fillna(market_const).to_numpy(dtype=float)
    momentum_score = np.tanh(momentum_pred / (np.nanstd(y_train) + 1e-6))
    base["simple_momentum"] = (momentum_score, momentum_pred)

    x_train = train["Headline_Count"].to_numpy(dtype=float)
    x_test = test["Headline_Count"].to_numpy(dtype=float)
    x_mean = float(np.mean(x_train)) if len(x_train) else 0.0
    x_std = float(np.std(x_train)) if len(x_train) else 1.0
    x_std = x_std if x_std > 1e-9 else 1.0
    x_train_z = (x_train - x_mean) / x_std
    x_test_z = (x_test - x_mean) / x_std
    slope = float(np.dot(x_train_z, y_train) / (np.dot(x_train_z, x_train_z) + 1e-12))
    intercept = float(np.mean(y_train) - slope * np.mean(x_train_z))
    count_pred = intercept + slope * x_test_z
    count_score = np.tanh(count_pred / (np.nanstd(y_train) + 1e-6))
    base["headline_count"] = (count_score, count_pred)

    return base


def _regime_map(train):
    daily = train.groupby("Date", as_index=False)["Actual_Pct_Change"].mean()
    if daily.empty:
        return {}
    q1 = daily["Actual_Pct_Change"].quantile(0.33)
    q2 = daily["Actual_Pct_Change"].quantile(0.67)
    mapping = {}
    for row in daily.itertuples(index=False):
        value = row.Actual_Pct_Change
        if value <= q1:
            mapping[row.Date] = "BEAR_REGIME"
        elif value >= q2:
            mapping[row.Date] = "BULL_REGIME"
        else:
            mapping[row.Date] = "SIDEWAYS_REGIME"
    return mapping


def run_research_evaluation(train_ratio=0.7, seed=42):
    """
    Full research evaluator with:
    - strong baselines
    - strict time split
    - significance testing
    - robustness checks
    - reproducibility manifest
    """
    df = _join_history()
    if df.empty:
        print("Historical data not available. Run pipeline for multiple days first.")
        return False

    train, test, cutoff = _time_split(df, train_ratio=train_ratio)
    if train.empty or test.empty:
        print("Not enough date coverage for strict time-split validation.")
        return False

    y_train = train["Actual_Pct_Change"].to_numpy(dtype=float)
    y_test = test["Actual_Pct_Change"].to_numpy(dtype=float)
    score_train = train["Pred_Score"].to_numpy(dtype=float)
    score_test = test["Pred_Score"].to_numpy(dtype=float)

    k_model = _calibrate_k(score_train, y_train)
    model_pred_pct = score_test * k_model
    results = []

    main_metrics = _compute_metrics(y_test, model_pred_pct, score_test, threshold=0.10)
    main_metrics.update(
        {
            "model": "sentiment_continuous",
            "k_value": k_model,
            "train_rows": len(train),
            "test_rows": len(test),
            "cutoff_date": str(cutoff.date()),
        }
    )
    results.append(main_metrics)

    baseline_preds = _baseline_predictions(train, test, seed=seed)
    for name, values in baseline_preds.items():
        b_score, b_pred_pct = values
        metrics = _compute_metrics(y_test, b_pred_pct, b_score, threshold=0.10)
        metrics.update({"model": name, "k_value": np.nan, "train_rows": len(train), "test_rows": len(test), "cutoff_date": str(cutoff.date())})
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("rmse")
    ensure_dirs()
    summary_path = os.path.join(RESEARCH_DIR, "research_summary.csv")
    results_df.to_csv(summary_path, index=False)

    # Robustness 1: sector-wise
    sector_rows = []
    for sector, block in test.groupby("Sector"):
        if len(block) < 8:
            continue
        sector_score = block["Pred_Score"].to_numpy(dtype=float)
        sector_y = block["Actual_Pct_Change"].to_numpy(dtype=float)
        sector_pred = sector_score * k_model
        sector_metrics = _compute_metrics(sector_y, sector_pred, sector_score, threshold=0.10)
        sector_metrics["slice_type"] = "sector"
        sector_metrics["slice_value"] = str(sector)
        sector_metrics["rows"] = len(block)
        sector_rows.append(sector_metrics)

    # Robustness 2: market regime-wise
    regime_lookup = _regime_map(train)
    test_with_regime = test.copy()
    test_with_regime["Regime"] = test_with_regime["Date"].map(regime_lookup).fillna("UNSEEN_REGIME")
    for regime, block in test_with_regime.groupby("Regime"):
        if len(block) < 8:
            continue
        regime_score = block["Pred_Score"].to_numpy(dtype=float)
        regime_y = block["Actual_Pct_Change"].to_numpy(dtype=float)
        regime_pred = regime_score * k_model
        regime_metrics = _compute_metrics(regime_y, regime_pred, regime_score, threshold=0.10)
        regime_metrics["slice_type"] = "market_regime"
        regime_metrics["slice_value"] = str(regime)
        regime_metrics["rows"] = len(block)
        sector_rows.append(regime_metrics)

    # Robustness 3: threshold sweep
    threshold_rows = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
        threshold_metrics = _compute_metrics(y_test, model_pred_pct, score_test, threshold=threshold)
        threshold_metrics["slice_type"] = "threshold"
        threshold_metrics["slice_value"] = threshold
        threshold_metrics["rows"] = len(test)
        threshold_rows.append(threshold_metrics)

    robustness_df = pd.DataFrame(sector_rows + threshold_rows)
    robustness_path = os.path.join(RESEARCH_DIR, "research_robustness.csv")
    robustness_df.to_csv(robustness_path, index=False)

    # Sponsored vs non-sponsored significance comparison
    split_rows = []
    for score_col, name in [("Sponsored_Pred_Score", "sponsored"), ("NonSponsored_Pred_Score", "non_sponsored")]:
        if score_col not in test.columns:
            continue
        subset = test.dropna(subset=[score_col]).copy()
        if len(subset) < 8:
            continue
        subset_score = subset[score_col].to_numpy(dtype=float)
        subset_y = subset["Actual_Pct_Change"].to_numpy(dtype=float)
        subset_k = _calibrate_k(subset_score, subset_y)
        subset_pred = subset_score * subset_k
        subset_metrics = _compute_metrics(subset_y, subset_pred, subset_score, threshold=0.10)
        subset_metrics["subset"] = name
        subset_metrics["rows"] = len(subset)
        split_rows.append(subset_metrics)
    split_df = pd.DataFrame(split_rows)
    split_path = os.path.join(RESEARCH_DIR, "research_sponsored_vs_non_sponsored.csv")
    split_df.to_csv(split_path, index=False)

    # Reproducibility manifest
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "seed": seed,
        "train_ratio": train_ratio,
        "cutoff_date": str(cutoff.date()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "input_hashes": {
            "nifty_sentiment_history.csv": _file_sha256(os.path.join(HISTORY_DIR, "nifty_sentiment_history.csv")),
            "daily_prices_history.csv": _file_sha256(os.path.join(HISTORY_DIR, "daily_prices_history.csv")),
        },
        "outputs": [
            os.path.basename(summary_path),
            os.path.basename(robustness_path),
            os.path.basename(split_path),
        ],
    }
    manifest_path = os.path.join(RESEARCH_DIR, "research_repro_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Research evaluation complete.")
    print(f"Saved: {summary_path}")
    print(f"Saved: {robustness_path}")
    print(f"Saved: {split_path}")
    print(f"Saved: {manifest_path}")
    return True


if __name__ == "__main__":
    run_research_evaluation()
