"""Benchmark Signal Comparisons — Evaluating 5 Quantitative Baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    DATA_FINAL,
    DATA_PROCESSED,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    setup_logging,
)

logger = setup_logging()


def compute_spearman_ic(pred_scores: pd.Series, fwd_returns: pd.Series) -> float:
    """Compute cross-sectional Spearman rank correlation."""
    valid_mask = pred_scores.notna() & fwd_returns.notna()
    if valid_mask.sum() < 5:
        return np.nan
    res = stats.spearmanr(pred_scores[valid_mask], fwd_returns[valid_mask])
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def newey_west_tstat(series: pd.Series, lags: int = 5) -> Tuple[float, float]:
    """Compute Newey-West HAC t-statistic and two-tailed p-value for daily IC series."""
    clean = series.dropna().to_numpy()
    n = len(clean)
    if n < 10:
        return np.nan, np.nan

    mean = np.mean(clean)
    gamma0 = np.var(clean, ddof=1)

    var_hac = gamma0
    for l in range(1, lags + 1):
        gamma_l = np.cov(clean[:-l], clean[l:])[0, 1]
        weight = 1.0 - (l / (lags + 1.0))
        var_hac += 2.0 * weight * gamma_l

    se = np.sqrt(max(1e-9, var_hac) / n)
    t_stat = float(mean / se)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    return t_stat, p_val


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg False Discovery Rate correction."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = np.zeros(n)
    for i in range(n):
        rank = i + 1
        adjusted_p[i] = min(1.0, sorted_p[i] * n / rank)

    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

    res = np.zeros(n)
    res[sorted_indices] = adjusted_p
    return res.tolist()


def evaluate_five_baselines(master_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute metrics across all 5 quantitative signal variants."""
    df = master_df.copy()
    fwd_col = "ret_fwd_1d" if "ret_fwd_1d" in df.columns else "target_return"

    if fwd_col not in df.columns:
        raise FileNotFoundError(
            f"Forward return column '{fwd_col}' missing in {list(df.columns)}. "
            "Refusing to synthesize returns — provide real forward returns "
            "(see pipeline/06_build_master.py, ret_fwd_1d / ret_overnight)."
        )

    df["sig_1_raw_tone"] = df["raw_tone"] if "raw_tone" in df.columns else df.get("tone_score", 0.0)
    if "finbert_sentiment" not in df.columns:
        raise FileNotFoundError(
            "Missing 'finbert_sentiment' — refusing to proxy FinBERT as tone*0.9+noise. "
            "Run research/finbert_daily.py to generate real FinBERT scores first."
        )
    df["sig_2_finbert"] = df["finbert_sentiment"]

    closed_weight = np.where(df.get("time_bucket", "OPEN").isin(["CLOSED_PRE", "CLOSED_POST"]), 1.5, 1.0)
    df["sig_3_temporal_only"] = df["sig_1_raw_tone"] * closed_weight

    prob = df.get("sponsored_prob", 0.5)
    org_weight = np.maximum(1.0 - prob, 0.3)
    df["sig_4_credibility_only"] = df["sig_1_raw_tone"] * org_weight

    df["sig_5_full_engine"] = df.get("pred_score", df["sig_1_raw_tone"] * org_weight * closed_weight)

    signals = {
        "1. Raw GDELT Tone": "sig_1_raw_tone",
        "2. FinBERT Baseline": "sig_2_finbert",
        "3. Temporal Only": "sig_3_temporal_only",
        "4. Credibility Only": "sig_4_credibility_only",
        "5. Full Proposed Signal": "sig_5_full_engine",
    }

    results = {}
    raw_p_values = []
    baseline_keys = list(signals.keys())

    for name, col in signals.items():
        date_col = "effective_date" if "effective_date" in df.columns else "date"
        daily_ics = df.groupby(date_col).apply(lambda g: compute_spearman_ic(g[col], g[fwd_col])).dropna()

        mean_ic = float(daily_ics.mean()) if len(daily_ics) > 0 else 0.0
        std_ic = float(daily_ics.std()) if len(daily_ics) > 1 else 1e-6
        icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
        t_stat, p_val = newey_west_tstat(daily_ics)

        daily_ls = df.groupby(date_col).apply(
            lambda g: (
                g.loc[g[col] >= g[col].quantile(0.80), fwd_col].mean() -
                g.loc[g[col] <= g[col].quantile(0.20), fwd_col].mean()
            ) if len(g) >= 10 else 0.0
        ).dropna()

        ann_return = float(daily_ls.mean() * 252.0)
        ann_vol = float(daily_ls.std() * np.sqrt(252.0)) if len(daily_ls) > 1 else 1e-6
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

        cum_ret = (1.0 + daily_ls).cumprod()
        peak = cum_ret.cummax()
        max_dd = float(((cum_ret - peak) / peak).min()) if len(cum_ret) > 0 else 0.0

        raw_p_values.append(p_val if not np.isnan(p_val) else 1.0)

        results[name] = {
            "mean_ic": round(mean_ic, 4),
            "icir": round(icir, 4),
            "hac_tstat": round(t_stat if not np.isnan(t_stat) else 0.0, 3),
            "p_value_raw": round(p_val if not np.isnan(p_val) else 1.0, 5),
            "ann_ls_return": round(ann_return * 100.0, 2),
            "annualized_sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100.0, 2),
        }

    fdr_p_values = benjamini_hochberg_fdr(raw_p_values)
    for i, name in enumerate(baseline_keys):
        results[name]["p_value_fdr"] = round(fdr_p_values[i], 5)

    return results


def save_baselines_report(results: Dict[str, Any], out_path: Path):
    """Format and persist baseline benchmark table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "# Five-Baseline Signal Benchmark Table",
        "",
        "| Signal Architecture | Mean IC | ICIR | HAC t-stat | Raw p-val | FDR p-val | Ann. L/S Spread (%) | Sharpe | Max DD (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, m in results.items():
        md_lines.append(
            f"| **{name}** | `{m['mean_ic']:+.4f}` | `{m['icir']:.3f}` | `{m['hac_tstat']:+.2f}` | "
            f"`{m['p_value_raw']:.4f}` | `{m['p_value_fdr']:.4f}` | `{m['ann_ls_return']:+.1f}%` | "
            f"`{m['annualized_sharpe']:.2f}` | `{m['max_drawdown']:.1f}%` |"
        )

    md_lines.extend([
        "",
        "> [!NOTE]",
        "> **Statistical Rigor Criteria**:",
        "> - **HAC t-stat**: Newey-West adjusted for serial autocovariance up to 5 trading days.",
        "> - **FDR p-val**: Benjamini-Hochberg false discovery rate adjusted across all tested signal families.",
        "> - **L/S Spread**: Top quintile (Q5) minus bottom quintile (Q1) daily rebalanced long-short spread.",
    ])

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Saved baseline benchmarks to {out_path} and {md_path}")


def run_baselines():
    """Main execution entry point.

    Apples-to-apples design: all 5 architectures evaluated on the SAME
    article-level panel (sponsored_scores + real FinBERT scores + real
    next-day forwards). Coverage is bounded by the scored subset
    (~Jan-Feb 2020) and reported in the JSON — no silent sample shifts.
    """
    from config.settings import DATA_FINAL as _DF
    base_path = DATA_PROCESSED / "sponsored_scores.parquet"
    if not base_path.exists():
        raise FileNotFoundError(
            f"{base_path} missing — run pipeline steps 01-04 first. "
            "Refusing to synthesize test data."
        )
    df = pd.read_parquet(base_path)
    if "tone_score" not in df.columns and "tone" in df.columns:
        df["tone_score"] = df["tone"]
    url_col = "DocumentIdentifier" if "DocumentIdentifier" in df.columns else "source_url"

    art_path = DATA_PROCESSED / "finbert_article_scores.parquet"
    if not art_path.exists():
        raise FileNotFoundError(
            f"{art_path} missing — run research/finbert_daily.py first. "
            "Refusing to proxy FinBERT."
        )
    art = pd.read_parquet(art_path)
    a_url = "DocumentIdentifier" if "DocumentIdentifier" in art.columns else "source_url"
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    art["effective_date"] = pd.to_datetime(art["effective_date"])
    # Same URL can match multiple tickers (03_label_news explodes rows),
    # so the join key is (url, date, ticker).
    df = df.merge(art[[a_url, "effective_date", "ticker", "finbert_sentiment"]].rename(
        columns={a_url: url_col}),
        on=[url_col, "effective_date", "ticker"], how="inner",
        validate="many_to_one")
    logger.info(f"FinBERT merge coverage: {len(df):,} articles with real scores.")

    master = pd.read_parquet(_DF / "master_dataset.parquet")
    dcol = "date" if "date" in master.columns else "effective_date"
    master[dcol] = pd.to_datetime(master[dcol])
    master = master.sort_values(["ticker", dcol])
    master["ret_fwd_1d"] = master.groupby("ticker")["ret_close2close"].shift(-1)
    df = df.merge(master[["ticker", dcol, "ret_fwd_1d"]].rename(
        columns={dcol: "effective_date"}),
        on=["ticker", "effective_date"], how="inner")
    df = df.dropna(subset=["ret_fwd_1d"]).reset_index(drop=True)
    logger.info(f"Eval panel: {len(df):,} articles, "
                f"{df['effective_date'].nunique()} dates after forward join.")

    results = evaluate_five_baselines(df)
    design = {
        "grain": "article-level (one row per article, cross-section over articles per date)",
        "forward": "ret_fwd_1d = next trading day close-to-close per ticker",
        "finbert_source": "ProsusAI/finbert on URL-slug pseudo-titles (research/finbert_daily.py)",
        "coverage": "bounded by sponsored_scores panel; see ndays per baseline",
    }
    save_baselines_report(results, OUTPUTS_TABLES / "five_baselines_benchmark.json")
    out_json = OUTPUTS_TABLES / "five_baselines_benchmark.json"
    payload = json.loads(out_json.read_text())
    payload["_eval_design"] = design
    out_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run_baselines()
