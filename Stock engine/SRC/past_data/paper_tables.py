import os
from datetime import datetime

import numpy as np
import pandas as pd

from SRC.common.paths import PAPER_DIR, RESEARCH_DIR, ensure_dirs

def _read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value, digits=4):
    try:
        number = float(value)
    except Exception:
        return "NA"
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def _to_markdown_table(df, output_path):
    if df.empty:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("No data available.\n")
        return

    header = "| " + " | ".join(df.columns) + " |\n"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    rows = []
    for row in df.itertuples(index=False):
        values = [str(getattr(row, col)) for col in df.columns]
        rows.append("| " + " | ".join(values) + " |\n")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(divider)
        handle.writelines(rows)


def _build_main_table(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    view = summary_df.copy()
    preferred_order = [
        "sentiment_continuous",
        "simple_momentum",
        "headline_count",
        "market_return",
        "random",
    ]
    view["model_rank"] = view["model"].apply(lambda x: preferred_order.index(x) if x in preferred_order else 999)
    view = view.sort_values(["model_rank", "rmse"]).drop(columns=["model_rank"])

    output = pd.DataFrame(
        {
            "Model": view["model"],
            "RMSE": view["rmse"].apply(_fmt),
            "MAE": view["mae"].apply(_fmt),
            "MAPE(%)": view["mape"].apply(_fmt, digits=2),
            "Pearson": view["pearson_corr"].apply(_fmt),
            "Pearson p": view["pearson_p"].apply(_fmt),
            "Spearman": view["spearman_corr"].apply(_fmt),
            "Spearman p": view["spearman_p"].apply(_fmt),
            "Dir Acc": view["directional_accuracy"].apply(_fmt),
            "Dir p": view["directional_pvalue"].apply(_fmt),
            "Test Rows": view["test_rows"].fillna(0).astype(int),
            "Cutoff Date": view["cutoff_date"].fillna("NA"),
        }
    )
    return output


def _build_robustness_table(robust_df):
    if robust_df.empty:
        return pd.DataFrame()

    keep_cols = ["slice_type", "slice_value", "rows", "rmse", "mae", "pearson_corr", "directional_accuracy"]
    data = robust_df[[col for col in keep_cols if col in robust_df.columns]].copy()
    if "slice_type" in data.columns:
        data = data.sort_values(["slice_type", "rmse"], ascending=[True, True])

    output = pd.DataFrame(
        {
            "Slice Type": data.get("slice_type", pd.Series(dtype=str)).fillna("NA"),
            "Slice Value": data.get("slice_value", pd.Series(dtype=str)).fillna("NA"),
            "Rows": data.get("rows", pd.Series(dtype=float)).fillna(0).astype(int),
            "RMSE": data.get("rmse", pd.Series(dtype=float)).apply(_fmt),
            "MAE": data.get("mae", pd.Series(dtype=float)).apply(_fmt),
            "Pearson": data.get("pearson_corr", pd.Series(dtype=float)).apply(_fmt),
            "Dir Acc": data.get("directional_accuracy", pd.Series(dtype=float)).apply(_fmt),
        }
    )
    return output


def _build_sponsored_table(split_df):
    if split_df.empty:
        return pd.DataFrame()

    view = split_df.copy()
    output = pd.DataFrame(
        {
            "Subset": view.get("subset", pd.Series(dtype=str)).fillna("NA"),
            "Rows": view.get("rows", pd.Series(dtype=float)).fillna(0).astype(int),
            "RMSE": view.get("rmse", pd.Series(dtype=float)).apply(_fmt),
            "MAE": view.get("mae", pd.Series(dtype=float)).apply(_fmt),
            "Pearson": view.get("pearson_corr", pd.Series(dtype=float)).apply(_fmt),
            "Pearson p": view.get("pearson_p", pd.Series(dtype=float)).apply(_fmt),
            "Dir Acc": view.get("directional_accuracy", pd.Series(dtype=float)).apply(_fmt),
            "Dir p": view.get("directional_pvalue", pd.Series(dtype=float)).apply(_fmt),
        }
    )
    return output


def _build_threshold_table(robust_df):
    if robust_df.empty or "slice_type" not in robust_df.columns:
        return pd.DataFrame()

    threshold_df = robust_df[robust_df["slice_type"] == "threshold"].copy()
    if threshold_df.empty:
        return pd.DataFrame()

    threshold_df = threshold_df.sort_values("slice_value")
    output = pd.DataFrame(
        {
            "Threshold": threshold_df["slice_value"].apply(_fmt, digits=2),
            "Rows": threshold_df.get("rows", pd.Series(dtype=float)).fillna(0).astype(int),
            "RMSE": threshold_df.get("rmse", pd.Series(dtype=float)).apply(_fmt),
            "MAE": threshold_df.get("mae", pd.Series(dtype=float)).apply(_fmt),
            "Pearson": threshold_df.get("pearson_corr", pd.Series(dtype=float)).apply(_fmt),
            "Dir Acc": threshold_df.get("directional_accuracy", pd.Series(dtype=float)).apply(_fmt),
            "Dir p": threshold_df.get("directional_pvalue", pd.Series(dtype=float)).apply(_fmt),
        }
    )
    return output


def generate_paper_tables():
    """
    Generate paper-ready tables from research outputs.

    Required input files:
    - research_summary.csv
    - research_robustness.csv
    - research_sponsored_vs_non_sponsored.csv
    """
    ensure_dirs()
    summary_df = _read_csv(os.path.join(RESEARCH_DIR, "research_summary.csv"))
    robust_df = _read_csv(os.path.join(RESEARCH_DIR, "research_robustness.csv"))
    split_df = _read_csv(os.path.join(RESEARCH_DIR, "research_sponsored_vs_non_sponsored.csv"))

    main_table = _build_main_table(summary_df)
    robust_table = _build_robustness_table(robust_df)
    sponsored_table = _build_sponsored_table(split_df)
    threshold_table = _build_threshold_table(robust_df)

    outputs = [
        (os.path.join(PAPER_DIR, "paper_table_main.csv"), main_table),
        (os.path.join(PAPER_DIR, "paper_table_robustness.csv"), robust_table),
        (os.path.join(PAPER_DIR, "paper_table_sponsored_vs_non_sponsored.csv"), sponsored_table),
        (os.path.join(PAPER_DIR, "paper_table_threshold_sweep.csv"), threshold_table),
    ]

    for path, frame in outputs:
        frame.to_csv(path, index=False)

    _to_markdown_table(main_table, os.path.join(PAPER_DIR, "paper_table_main.md"))
    _to_markdown_table(robust_table, os.path.join(PAPER_DIR, "paper_table_robustness.md"))
    _to_markdown_table(sponsored_table, os.path.join(PAPER_DIR, "paper_table_sponsored_vs_non_sponsored.md"))
    _to_markdown_table(threshold_table, os.path.join(PAPER_DIR, "paper_table_threshold_sweep.md"))

    timestamp = datetime.utcnow().isoformat()
    manifest_path = os.path.join(PAPER_DIR, "paper_tables_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write("Paper Tables Generation Manifest\n")
        handle.write(f"Generated UTC: {timestamp}\n")
        handle.write(f"Input research_summary.csv rows: {len(summary_df)}\n")
        handle.write(f"Input research_robustness.csv rows: {len(robust_df)}\n")
        handle.write(f"Input research_sponsored_vs_non_sponsored.csv rows: {len(split_df)}\n")
        handle.write("Outputs:\n")
        handle.write("- paper_table_main.csv\n")
        handle.write("- paper_table_main.md\n")
        handle.write("- paper_table_robustness.csv\n")
        handle.write("- paper_table_robustness.md\n")
        handle.write("- paper_table_sponsored_vs_non_sponsored.csv\n")
        handle.write("- paper_table_sponsored_vs_non_sponsored.md\n")
        handle.write("- paper_table_threshold_sweep.csv\n")
        handle.write("- paper_table_threshold_sweep.md\n")
        handle.write("- paper_tables_manifest.txt\n")

    print("Paper tables generated.")
    print(f"Saved: {PAPER_DIR}")
    return True


if __name__ == "__main__":
    generate_paper_tables()
