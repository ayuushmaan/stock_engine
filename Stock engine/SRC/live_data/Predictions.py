from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytz
from scipy import stats

from SRC.common.model_config import load_trained_parameters
from SRC.common.paths import HISTORY_DIR, LATEST_DIR, ensure_dirs

def _safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _compute_calibration_k(pred_score: np.ndarray, actual_pct: np.ndarray) -> float:
    denominator = float(np.dot(pred_score, pred_score))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(pred_score, actual_pct) / denominator)


def generate_prediction_report():
    """
    Generate continuous-score prediction report.

    Primary output:
    - Pred_Score in [-1, 1]
    - Expected_Change_Pct calibrated from current run
    """
    ist = pytz.timezone("Asia/Kolkata")
    print("Loading Data Files...\n")

    ensure_dirs()
    prices_path = os.path.join(LATEST_DIR, "daily_prices.csv")
    sentiment_path = os.path.join(LATEST_DIR, "nifty_sentiment_results.csv")

    if not os.path.exists(prices_path):
        print(f"{prices_path} not found.")
        print("Please run: python app.py prices")
        return False

    if not os.path.exists(sentiment_path):
        print(f"{sentiment_path} not found.")
        print("Please run sentiment analysis first.")
        return False

    prices = pd.read_csv(prices_path)
    sentiment = pd.read_csv(sentiment_path)
    print(f"Loaded {prices_path} ({len(prices)} stocks)")
    print(f"Loaded {sentiment_path} ({len(sentiment)} signals)\n")

    merged = pd.merge(prices, sentiment, on=["Symbol"], how="inner", suffixes=("_price", "_sent"))
    if merged.empty:
        print("No common stocks found between prices and sentiment files.")
        return False

    if "Date_price" in merged.columns:
        merged["Date"] = merged["Date_price"]
    elif "Date_sent" in merged.columns:
        merged["Date"] = merged["Date_sent"]
    elif "Date" not in merged.columns:
        merged["Date"] = datetime.now(ist).date().isoformat()

    if "Pred_Score" not in merged.columns:
        merged["Pred_Score"] = merged.get("Sentiment_Score", 0.0)
    merged["Pred_Score"] = merged["Pred_Score"].map(_safe_float).fillna(0.0).clip(-1.0, 1.0)
    merged["Percent_Change"] = merged["Percent_Change"].map(_safe_float).fillna(0.0)

    pred_values = merged["Pred_Score"].to_numpy(dtype=float)
    actual_values = merged["Percent_Change"].to_numpy(dtype=float)
    trained_parameters = load_trained_parameters()
    if trained_parameters.calibration_k:
        calibration_k = trained_parameters.calibration_k
        calibration_source = "historical_training"
    else:
        calibration_k = _compute_calibration_k(pred_values, actual_values)
        calibration_source = "current_batch_fallback"
    merged["Expected_Change_Pct"] = merged["Pred_Score"] * calibration_k
    merged["Calibration_K_Used"] = calibration_k
    merged["Calibration_Source"] = calibration_source

    merged["Pred_Direction"] = np.where(
        merged["Pred_Score"] > 0.10,
        "UP",
        np.where(merged["Pred_Score"] < -0.10, "DOWN", "NEUTRAL"),
    )
    merged["Correct"] = np.where(
        merged["Pred_Direction"] == "NEUTRAL",
        "SKIPPED",
        np.where(merged["Pred_Direction"] == merged["Actual_Direction"], "CORRECT", "WRONG"),
    )

    directional_mask = merged["Pred_Direction"] != "NEUTRAL"
    directional_total = int(directional_mask.sum())
    directional_correct = int((merged.loc[directional_mask, "Correct"] == "CORRECT").sum())
    directional_accuracy = (directional_correct / directional_total * 100.0) if directional_total else np.nan

    mae = float(np.mean(np.abs(merged["Expected_Change_Pct"] - merged["Percent_Change"])))
    rmse = float(np.sqrt(np.mean((merged["Expected_Change_Pct"] - merged["Percent_Change"]) ** 2)))
    mape_series = np.abs((merged["Expected_Change_Pct"] - merged["Percent_Change"]) / merged["Percent_Change"].replace(0, np.nan))
    mape = float(np.nanmean(mape_series) * 100) if np.isfinite(np.nanmean(mape_series)) else np.nan

    if len(merged) >= 3:
        pearson_corr, pearson_p = stats.pearsonr(merged["Pred_Score"], merged["Percent_Change"])
        spearman_corr, spearman_p = stats.spearmanr(merged["Pred_Score"], merged["Percent_Change"])
    else:
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan

    output_cols = [
        "Date",
        "Symbol",
        "Sector_price" if "Sector_price" in merged.columns else "Sector",
        "Headline_Count",
        "Sponsored_Count",
        "NonSponsored_Count",
        "Pred_Score",
        "Direction",
        "Pred_Direction",
        "Expected_Change_Pct",
        "Calibration_K_Used",
        "Calibration_Source",
        "Percent_Change",
        "Actual_Direction",
        "Correct",
    ]
    output_cols = [col for col in output_cols if col in merged.columns]
    results_df = merged[output_cols].copy()
    if "Sector_price" in results_df.columns:
        results_df = results_df.rename(columns={"Sector_price": "Sector"})

    latest_report_path = os.path.join(LATEST_DIR, "prediction_report.csv")
    results_df.to_csv(latest_report_path, index=False)

    history_path = os.path.join(HISTORY_DIR, "prediction_report_history.csv")
    if pd.io.common.file_exists(history_path):
        existing = pd.read_csv(history_path)
        history = pd.concat([existing, results_df], ignore_index=True)
        history = history.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    else:
        history = results_df.copy()
    history.to_csv(history_path, index=False)

    print("=" * 95)
    print(f"{'SYMBOL':<12} | {'SCORE':<8} | {'PRED':<8} | {'EXPECTED%':<10} | {'ACTUAL%':<10} | {'RESULT':<8}")
    print("-" * 95)
    for row in results_df.itertuples(index=False):
        print(
            f"{row.Symbol:<12} | "
            f"{getattr(row, 'Pred_Score'):>6.3f} | "
            f"{getattr(row, 'Pred_Direction'):<8} | "
            f"{getattr(row, 'Expected_Change_Pct'):>8.3f} | "
            f"{getattr(row, 'Percent_Change'):>8.3f} | "
            f"{getattr(row, 'Correct'):<8}"
        )
    print("=" * 95)

    print("\nPREDICTION ANALYSIS SUMMARY")
    print(f"Samples:                    {len(results_df)}")
    print(f"Directional predictions:    {directional_total}")
    print(f"Directional accuracy:       {directional_accuracy:.2f}%" if np.isfinite(directional_accuracy) else "Directional accuracy:       N/A")
    print(f"Calibration k:              {calibration_k:.4f}")
    print(f"Calibration source:         {calibration_source}")
    print(f"MAE (%):                    {mae:.4f}")
    print(f"RMSE (%):                   {rmse:.4f}")
    print(f"MAPE (%):                   {mape:.4f}" if np.isfinite(mape) else "MAPE (%):                   N/A")
    print(f"Pearson corr (p-value):     {pearson_corr:.4f} ({pearson_p:.4g})" if np.isfinite(pearson_corr) else "Pearson corr (p-value):     N/A")
    print(f"Spearman corr (p-value):    {spearman_corr:.4f} ({spearman_p:.4g})" if np.isfinite(spearman_corr) else "Spearman corr (p-value):    N/A")
    print(f"Report generated:           {datetime.now(ist).strftime('%d/%m/%Y %H:%M:%S IST')}")

    print(f"\nDetailed report saved to '{latest_report_path}'")
    print(f"History updated in '{history_path}'")
    return True


if __name__ == "__main__":
    generate_prediction_report()
